"""IDS706: sleep quality, cognition, age, and education in RAND HRS.

Run stages 1–4 independently with --stage; revised results use output/research_v2/.
Sleep categories are researcher-defined. Associations are noncausal.
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from importlib.metadata import version

from pyspark.sql import SparkSession, functions as F

WAVES = range(6, 17)
SLEEP = ("sleepfal", "sleepwkn", "sleepwke", "sleeprt")
SCHEMA_VERSION = "research_v2"
EDUCATION = {1: "Less than high school", 2: "GED", 3: "High-school graduate",
             4: "Some college", 5: "College and above"}
ACTIVE_STAGE = 0
ROOT = None


def progress(message):
    text = f"[Stage {ACTIVE_STAGE}] {message}"
    print(text, flush=True)
    if ROOT is not None:
        with (ROOT / "execution_log.txt").open("a") as stream:
            stream.write(text + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + "\n")


def convert_selected(source, root, chunk_size):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyreadstat

    _, metadata = pyreadstat.read_dta(source, metadataonly=True)
    columns = ["hhid", "pn", "hhidpn", "raeduc"]
    for wave in WAVES:
        suffixes = ["agey_b", "iwstat", "proxy", "iwendy", *SLEEP]
        for mode in ([""] if wave < 14 else ["p", "w"]):
            suffixes.extend(f"{name}{mode}" for name in ("tr20", "fimrc", "fdlrc"))
        columns.extend(f"r{wave}{suffix}" for suffix in suffixes)
    missing = set(columns) - set(metadata.column_names)
    require(not missing, f"Required RAND variables are missing: {sorted(missing)}")
    for wave in WAVES:
        for suffix in SLEEP:
            labels = metadata.variable_value_labels.get(f"r{wave}{suffix}", {})
            require(all(code in labels for code in (1, 2, 3)),
                    f"Unexpected sleep codes in wave {wave}.")

    require(set(range(1, 6)) <= set(metadata.variable_value_labels["raeduc"]),
            "Unexpected education coding.")
    schema = pa.schema([
        pa.field(name, pa.string() if name in ("hhid", "pn") else
                 pa.int64() if name == "hhidpn" else pa.int16())
        for name in columns
    ])
    destination = root / "input_selected"
    destination.mkdir()
    rows = 0
    # Load: convert selected columns in bounded chunks.
    chunks = pyreadstat.read_file_in_chunks(
        pyreadstat.read_dta, source, chunksize=chunk_size, multiprocess=False,
        num_rows=metadata.number_rows, usecols=columns, apply_value_formats=False,
        user_missing=False, disable_datetime_conversion=True,
    )
    for index, (frame, _) in enumerate(chunks):
        # Safe casts preserve exact IDs and stable chunk types.
        arrays = [pa.array(frame[field.name], type=field.type, from_pandas=True,
                           safe=True) for field in schema]
        table = pa.Table.from_arrays(arrays, schema=schema)
        require(table.column("hhidpn").null_count == 0, "Missing respondent ID.")
        pq.write_table(table, destination / f"part-{index:05d}.parquet",
                       compression="snappy", row_group_size=chunk_size)
        rows += table.num_rows
        progress(f"Converted {rows:,}/{metadata.number_rows:,} respondents")
    require(rows == metadata.number_rows, "Conversion row count differs from Stata metadata.")
    footers = [pq.ParquetFile(p).metadata.serialized_size
               for p in sorted(destination.glob("*.parquet"))]
    return {
        "source_rows": metadata.number_rows, "source_columns": metadata.number_columns,
        "selected_columns": columns, "converted_rows": rows,
        "selected_schema_columns": len(schema), "footer_bytes_per_file": footers,
        "variable_labels": {n: metadata.column_names_to_labels[n] for n in columns},
        "value_labels": {n: metadata.variable_value_labels[n] for n in columns
                         if n in metadata.variable_value_labels},
    }



# SQL 1: descriptive comparisons within age, education, and wave strata.
GROUP_SQL = """
SELECT wave, age_band, raeduc, sleep_quality,
       COUNT(*) AS observations, COUNT(DISTINCT hhidpn) AS respondents,
       AVG(age) AS mean_age, AVG(recall) AS mean_recall,
       STDDEV_SAMP(recall) AS sd_recall
FROM analysis
GROUP BY wave, age_band, raeduc, sleep_quality
"""
# SQL 2: CTE and self-join compare medium/low quality against high quality.
CONTRAST_SQL = """
WITH grouped AS (
 SELECT wave, age_band, raeduc, sleep_quality, COUNT(*) AS observations,
        AVG(recall) AS mean_recall
 FROM analysis GROUP BY wave, age_band, raeduc, sleep_quality
)
SELECT a.wave, a.age_band, a.raeduc, a.sleep_quality,
       a.observations, h.observations AS high_observations,
       a.mean_recall, h.mean_recall AS high_mean_recall,
       a.mean_recall-h.mean_recall AS difference_from_high
FROM grouped a JOIN grouped h
 ON a.wave=h.wave AND a.age_band=h.age_band AND a.raeduc=h.raeduc
WHERE h.sleep_quality='high' AND a.sleep_quality IN ('medium','low')
"""


def with_sleep_quality(frame):
    """Reverse rested coding; require all four valid items before summing."""
    valid = F.lit(True)
    for name in SLEEP:
        valid = valid & F.col(name).isin(1, 2, 3)
    score = sum((3-F.col(n) for n in SLEEP[:3]), F.lit(0)) + F.col("sleeprt")-1
    return (frame.withColumn("sleep_disturbance", F.when(valid, score.cast("int")))
            .withColumn("sleep_quality", F.when(F.col("sleep_disturbance") <= 2, "high")
                        .when(F.col("sleep_disturbance") <= 5, "medium")
                        .when(F.col("sleep_disturbance") <= 8, "low")))


def primary(panel):
    """Observed face-to-face/telephone recall, non-proxy, complete primary cohort."""
    return (panel.filter((F.col("proxy") == 0) & F.col("raeduc").isin(1,2,3,4,5))
            .filter(F.col("sleep_quality").isNotNull() & F.col("tr20_ftf_tel").between(0,20))
            .filter((F.col("fimrc_ftf_tel") == 0) & (F.col("fdlrc_ftf_tel") == 0))
            .select("hhidpn", "wave", "age", "age_band", "raeduc", "sleep_quality",
                    "sleep_disturbance", F.col("tr20_ftf_tel").cast("double").alias("recall")))


def read_analysis(args, manifest):
    source = Path(args.stage1_dir).resolve()
    saved = json.loads((source / "run_manifest.json").read_text())
    require(saved.get("schema_version") == SCHEMA_VERSION and saved.get("status") == "complete",
            "Use a complete research_v2 Stage 1 dataset.")
    manifest["stage1_manifest_sha256"] = sha256(source / "run_manifest.json")
    return source


def save_plans(frame, root, name):
    for mode in ("formatted", "extended"):
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            frame.explain(mode=mode)
        (root / "plans").mkdir(exist_ok=True)
        (root / "plans" / f"{name}_{mode}.txt").write_text(capture.getvalue())
        if mode == "formatted":
            progress(f"{name}:\n{capture.getvalue()}")


def save_parquet(frame, root, name, partitions=2):
    path = root / name
    frame.coalesce(partitions).write.mode("errorifexists").parquet(str(path))
    reread = frame.sparkSession.read.parquet(str(path))
    require([(f.name,f.dataType) for f in frame.schema] == [(f.name,f.dataType) for f in reread.schema],
            f"Parquet schema mismatch: {name}")
    return reread.count()


def rows_equal(actual, expected, keys):
    """Compare small summaries with floating-point tolerance."""
    a = {tuple(r[k] for k in keys): r for r in actual}
    b = {tuple(r[k] for k in keys): r for r in expected}
    require(len(a) == len(actual) and set(a) == set(b), "Summary keys differ.")
    for key in a:
        require(set(a[key]) == set(b[key]), "Summary columns differ.")
        for name,x in a[key].items():
            y = b[key][name]
            equal = math.isclose(x,y,rel_tol=1e-9,abs_tol=1e-9) if isinstance(x,(int,float)) and isinstance(y,(int,float)) else x == y
            require(equal, f"Summary mismatch: {key} {name}: {x} != {y}")


def run_stage1(args, spark, root, manifest):
    """Stage 1 — chunked ingestion, transformations, SQL, and Parquet outputs."""
    from pyspark import StorageLevel
    source = Path(args.input).resolve()
    manifest["source_bytes"] = source.stat().st_size
    manifest["source_sha256"] = sha256(source)
    manifest["rules"] = {"education": EDUCATION, "sleep_score": "sum(3-trouble_item) + rested-1; all four valid",
        "sleep_categories": {"high":[0,2], "medium":[3,5], "low":[6,8]},
        "clinical_scale": False, "primary": "age50+, nonproxy, education present, ftf/tel recall 0..20, both imputation flags=0"}
    manifest.update(convert_selected(source, root, args.chunk_size))
    wide = spark.read.parquet(str(root / "input_selected"))
    identity = wide.agg(F.count("*").alias("rows"), F.countDistinct("hhidpn").alias("ids"),
                        F.sum(F.when(F.concat("hhid","pn").cast("long") != F.col("hhidpn"),1).otherwise(0)).alias("bad")).first()
    require(identity.rows == identity.ids == manifest["source_rows"] and identity.bad == 0, "Respondent identifiers differ.")
    records = []
    for w in WAVES:
        fields = [F.lit(w).alias("wave"), F.col(f"r{w}agey_b").alias("age"),
                  F.col(f"r{w}iwstat").alias("interview_status"),F.col(f"r{w}proxy").alias("proxy")]
        fields += [F.col(f"r{w}{n}").alias(n) for n in SLEEP]
        for mode,suffix in (("ftf_tel", "" if w<14 else "p"),("web","w")):
            for n in ("tr20","fimrc","fdlrc"):
                value = F.col(f"r{w}{n}{suffix}") if mode=="ftf_tel" or w>=14 else F.lit(None).cast("short")
                fields.append(value.alias(f"{n}_{mode}"))
        records.append(F.struct(*fields))
    long = wide.select("hhidpn","raeduc", F.explode(F.array(*records)).alias("r")).select("hhidpn","raeduc","r.*")
    # Filter and withColumn requirements; missing codes remain null.
    participating = long.filter(F.col("interview_status").isin(1,2))
    cohort = participating.filter(F.col("age") >= 50)
    cohort = cohort.withColumn("raeduc", F.when(F.col("raeduc").isin(1,2,3,4,5),F.col("raeduc")))
    for n in SLEEP:
        cohort = cohort.withColumn(n,F.when(F.col(n).isin(1,2,3),F.col(n)))
    for mode in ("ftf_tel","web"):
        cohort = cohort.withColumn(f"tr20_{mode}",F.when(F.col(f"tr20_{mode}").between(0,20),F.col(f"tr20_{mode}")))
    panel = (with_sleep_quality(cohort).withColumn("age_band",F.when(F.col("age")<65,"50-64")
             .when(F.col("age")<75,"65-74").when(F.col("age")<85,"75-84").otherwise("85+"))
             .persist(StorageLevel.MEMORY_AND_DISK))
    require(panel.groupBy("hhidpn","wave").count().filter("count>1").count()==0,"Duplicate respondent-wave keys.")
    # Verify direction, thresholds, and null propagation independently on boundary examples.
    examples = [(3,3,3,1,0,"high"),(3,3,1,1,2,"high"),(3,3,1,2,3,"medium"),
                (2,2,1,2,5,"medium"),(1,1,2,2,6,"low"),(1,1,1,3,8,"low"),(None,3,3,1,None,None)]
    test = spark.createDataFrame([r[:4] for r in examples],list(SLEEP))
    for r,e in zip(with_sleep_quality(test).collect(),examples):
        require((r.sleep_disturbance,r.sleep_quality)==e[4:],"Sleep classification boundary check failed.")
    analysis = primary(panel).persist(StorageLevel.MEMORY_AND_DISK)
    analysis.createOrReplaceTempView("analysis")
    grouped, contrasts = spark.sql(GROUP_SQL), spark.sql(CONTRAST_SQL)
    keys=["wave","age_band","raeduc","sleep_quality"]
    independent = analysis.groupBy(*keys).agg(F.count("*").alias("observations"),F.countDistinct("hhidpn").alias("respondents"),
                  F.avg("age").alias("mean_age"),F.avg("recall").alias("mean_recall"),F.stddev_samp("recall").alias("sd_recall"))
    rows_equal([r.asDict() for r in grouped.collect()],[r.asDict() for r in independent.collect()],keys)
    descriptions = analysis.groupBy("sleep_quality").agg(F.count("*").alias("observations"),F.countDistinct("hhidpn").alias("respondents"),
                  F.avg("age").alias("mean_age"),F.avg("recall").alias("mean_recall"),F.stddev_samp("recall").alias("sd_recall"))
    exclusions=[];current=panel
    for name,rule in [("nonproxy",F.col("proxy")==0),("education",F.col("raeduc").isNotNull()),
                      ("sleep_complete",F.col("sleep_quality").isNotNull()),("recall",F.col("tr20_ftf_tel").isNotNull()),
                      ("observed_recall",(F.col("fimrc_ftf_tel")==0)&(F.col("fdlrc_ftf_tel")==0))]:
        before=current.count();current=current.filter(rule);after=current.count()
        exclusions.append({"step":name,"excluded":before-after,"retained":after})
    manifest["counts"]={"respondents":identity.ids,"participating_waves":participating.count(),"cleaned_waves":panel.count(),
        "primary_observations":analysis.count(),"primary_respondents":analysis.select("hhidpn").distinct().count()}
    manifest["exclusions"]=exclusions
    manifest["sleep_descriptives"]=[r.asDict() for r in descriptions.orderBy("sleep_quality").collect()]
    manifest["sparse_strata_under_20"] = grouped.filter("observations<20").count()
    # Column pruning, early filters, AQE and bounded writes avoid unnecessary shuffles.
    manifest["outputs"]={}
    for name,frame in [("respondent_wave",panel),("analysis",analysis),("sleep_recall_summary",grouped),
                       ("stratified_comparisons",contrasts),("sleep_descriptives",descriptions)]:
        n=save_parquet(frame,root,name)
        require(n==frame.count(),f"Read-back row count differs: {name}")
        manifest["outputs"][name]=n
    grouped.orderBy(*keys).show(12,truncate=False)
    contrasts.orderBy(*keys).show(8,truncate=False)
    require(sum(r["observations"] for r in manifest["sleep_descriptives"])==manifest["counts"]["primary_observations"],"Denominator mismatch.")
    require(sha256(source)==manifest["source_sha256"],"Original source changed.")
    manifest["verification"]={"sleep_boundaries":True,"sql_dataframe_agreement":True,"ids_exact":True,"source_unchanged":True}
    progress(f"Counts: {manifest['counts']}; sleep descriptives: {manifest['sleep_descriptives']}")
    analysis.unpersist();panel.unpersist()


def capture_ui(spark, root, query_jobs):
    """Capture genuine live Spark UI pages with existing headless Chrome."""
    ui=spark.sparkContext.uiWebUrl
    require(ui,"Spark UI unavailable.")
    api=f"{ui}/api/v1/applications/{spark.sparkContext.applicationId}"
    def fetch(url):
        with urllib.request.urlopen(url,timeout=30) as response:return response.read()
    evidence=root/"ui_evidence";evidence.mkdir()
    snapshots={}
    for endpoint in ("jobs","stages","sql","storage/rdd"):
        data=fetch(f"{api}/{endpoint}");(evidence/f"{endpoint.replace('/','_')}.json").write_bytes(data)
        snapshots[endpoint]=json.loads(data)
    stages=[r for r in snapshots["stages"] if r["status"]=="COMPLETE"]
    stage=max(stages,key=lambda r:r.get("shuffleReadBytes",0)+r.get("shuffleWriteBytes",0))
    ids={}
    for name,jobs in query_jobs.items():
        candidates=[q for q in snapshots["sql"] if set(q.get("successJobIds",[])) & set(jobs)]
        require(candidates,f"No UI query found for {name}")
        ids[name]=max(candidates,key=lambda q:q.get("duration",0))["id"]
    views=[("01_query_plan.png",f"{ui}/SQL/execution/?id={ids['grouped']}"),
           ("02_successful_execution.png",f"{ui}/stages/stage/?id={stage['stageId']}&attempt={stage['attemptId']}"),
           ("03_query_details.png",f"{ui}/SQL/execution/?id={ids['contrasts']}")]
    chrome=Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    require(chrome.is_file(),"Installed Chrome not found; no browser packages will be installed.")
    screenshots=root/"screenshots";screenshots.mkdir()
    records=[]
    for name,url in views:
        (evidence/name.replace('.png','.html')).write_bytes(fetch(url))
        command=[str(chrome),"--headless","--disable-gpu","--no-first-run","--no-default-browser-check",
            f"--user-data-dir={root/'chrome_profile'}","--hide-scrollbars","--window-size=1700,2600",
            "--virtual-time-budget=5000","--timeout=20000",f"--screenshot={screenshots/name}",url]
        result=subprocess.run(command,capture_output=True,text=True,timeout=60)
        (evidence/name.replace('.png','.log')).write_text(result.stdout+result.stderr)
        require(result.returncode==0 and (screenshots/name).is_file(),f"Screenshot failed: {name}")
        require((screenshots/name).read_bytes()[:8]==b'\x89PNG\r\n\x1a\n',"Invalid PNG.")
        records.append({"filename":name,"url":url,"sha256":sha256(screenshots/name)})
        progress(f"Captured {name}")
    return {"ui_url":ui,"screenshots":records,"selected_stage":stage,
            "storage":snapshots["storage/rdd"],"all_stages":stages,"query_ids":ids}


def run_stage2(args,spark,root,manifest):
    """Stage 2 — physical plans, repeated timings, cache costs, and UI evidence."""
    source=read_analysis(args,manifest)
    # Rebuild the primary filter from the cleaned panel to inspect Parquet pushdown.
    base=primary(spark.read.parquet(str(source/"respondent_wave")))
    require(base.count()==json.loads((source/"run_manifest.json").read_text())["counts"]["primary_observations"],"Primary cohort mismatch.")
    references={n:[r.asDict() for r in spark.read.parquet(str(source/p)).collect()]
                for n,p in [("grouped","sleep_recall_summary"),("contrasts","stratified_comparisons")]}
    sqls={"grouped":GROUP_SQL,"contrasts":CONTRAST_SQL}
    keys=["wave","age_band","raeduc","sleep_quality"]
    timings=[];query_jobs={}
    def trial(name,data,condition,parts,phase,i,plans=False):
        spark.conf.set("spark.sql.shuffle.partitions",parts)
        data.createOrReplaceTempView("analysis")
        frame=spark.sql(sqls[name]);label=f"{name}_{condition}_p{parts}"
        if plans:save_plans(frame,root,label+"_before")
        group=f"v2:{label}:{phase}:{i}";spark.sparkContext.setJobGroup(group,group)
        start=time.perf_counter();rows=[r.asDict() for r in frame.collect()];elapsed=time.perf_counter()-start
        jobs=list(spark.sparkContext.statusTracker().getJobIdsForGroup(group))
        rows_equal(rows,references[name],keys)
        timings.append({"workload":name,"condition":condition,"partitions":parts,"phase":phase,"iteration":i,"seconds":elapsed,"jobs":jobs})
        if plans:
            save_plans(frame,root,label+"_after")
            if condition=="uncached" and parts==4:query_jobs[name]=jobs
        progress(f"{label} {phase} {i}: {elapsed:.4f}s")
    for parts in (4,2):
        for name in sqls:
            if parts==4:trial(name,base,"uncached",parts,"first",0,True)
            trial(name,base,"uncached",parts,"warmup",0)
            for i in range(1,args.repeats+1):trial(name,base,"uncached",parts,"measured",i,parts==2 and i==1)
    spark.sparkContext.setJobGroup("v2:cache_population","Materialize shared primary cohort")
    cached=base.cache();start=time.perf_counter();n=cached.count();population=time.perf_counter()-start
    for name in sqls:
        trial(name,cached,"cached",4,"warmup",0,True)
        for i in range(1,args.repeats+1):trial(name,cached,"cached",4,"measured",i)
    with (root/"timings.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(timings[0]));writer.writeheader();writer.writerows(timings)
    summaries=[]
    for name in sqls:
        for condition,parts in [("uncached",4),("uncached",2),("cached",4)]:
            values=[t["seconds"] for t in timings if (t["workload"],t["condition"],t["partitions"],t["phase"])==(name,condition,parts,"measured")]
            summaries.append({"workload":name,"condition":condition,"partitions":parts,"median":statistics.median(values),
                              "min":min(values),"max":max(values),"total":sum(values),"repetitions":len(values)})
    manifest["timing_summaries"]=summaries
    manifest["cache"]={"population_seconds":population,"rows":n,
        "uncached_total":sum(t["seconds"] for t in timings if t["condition"]=="uncached" and t["partitions"]==4 and t["phase"]=="measured"),
        "cached_total_with_population":population+sum(t["seconds"] for t in timings if t["condition"]=="cached" and t["phase"]=="measured")}
    ui=capture_ui(spark,root,query_jobs)
    write_json(root/"ui_metrics.json",ui)
    manifest["screenshots"]=ui["screenshots"]
    plan=(root/"plans/grouped_uncached_p4_after_formatted.txt").read_text()
    contrast_plan=(root/"plans/contrasts_uncached_p4_after_formatted.txt").read_text()
    cached_plan=(root/"plans/grouped_cached_p4_after_formatted.txt").read_text()
    require("InMemoryTableScan" in cached_plan,"Cached scan not observed.")
    scan=[line.strip() for line in plan.splitlines() if any(t in line for t in ("PushedFilters:","ReadSchema:","PartitionFilters:"))]
    observed={"scan":scan,"coalesced": "coalesced" in plan,"join_operators":[x for x in ["BroadcastHashJoin","SortMergeJoin"] if x in contrast_plan],
        "task_counts":sorted({s["numCompleteTasks"] for s in ui["all_stages"]}),
        "max_shuffle_write_bytes":max(s.get("shuffleWriteBytes",0) for s in ui["all_stages"]),
        "memory_spill":sum(s.get("memoryBytesSpilled",0) for s in ui["all_stages"]),
        "disk_spill":sum(s.get("diskBytesSpilled",0) for s in ui["all_stages"])}
    manifest["plan_observations"]=observed
    paragraphs=["Revised primary-cohort queries filter and project before grouping. Actual scan details: "+" | ".join(scan)+
        ". Pushed predicates do not prove row-group skipping. The derived sleep category filter can remain above the scan. No directory partition pruning is claimed.",
        f"Grouped statistics require partial/final aggregates and Exchanges; the contrast query uses a CTE/self-join. Observed joins: {observed['join_operators']}. "
        f"AQE coalesced reads: {observed['coalesced']}; completed task counts {observed['task_counts']}. Largest stage shuffle write {observed['max_shuffle_write_bytes']} bytes; "
        f"memory/disk spill {observed['memory_spill']}/{observed['disk_spill']} bytes. Configured partitions are not executed task counts. These local measurements do not identify a general CPU/I/O bottleneck.",
        f"Five-trial summaries: {summaries}. Cache population: {population:.4f}s for {n} rows. Measured uncached batch {manifest['cache']['uncached_total']:.4f}s; "
        f"cached batch including population {manifest['cache']['cached_total_with_population']:.4f}s. First runs/warmups are separate; fixed order, OS cache, JIT and machine noise limit interpretation. "
        "Only Spark caching is controlled; this is not a cold-disk benchmark."]
    (root/"performance_analysis.txt").write_text("\n\n".join(paragraphs)+"\n")
    manifest["verification"]={"all_results_match_stage1":True,"cached_plan_verified":True,"screenshots_captured":True}
    cached.unpersist(blocking=True)


def run_stage3(args,spark,root,manifest):
    """Stage 3 — transformations define plans; actions submit jobs."""
    source=read_analysis(args,manifest);sc=spark.sparkContext;tracker=sc.statusTracker();evidence=[]
    def record(operation,category,group,fn):
        sc.setJobGroup(group,operation);before=sorted(tracker.getJobIdsForGroup(group));start=time.perf_counter()
        result=fn();elapsed=time.perf_counter()-start
        if category=="transformation":time.sleep(.1)
        deadline=time.perf_counter()+5;after=sorted(tracker.getJobIdsForGroup(group))
        while category=="action" and not set(after)-set(before) and time.perf_counter()<deadline:
            time.sleep(.05);after=sorted(tracker.getJobIdsForGroup(group))
        new=sorted(set(after)-set(before))
        if category=="transformation":require(not new,"Transformation submitted jobs.")
        if category=="action":require(new,"No action job evidence.")
        evidence.append({"operation":operation,"category":category,"job_group":group,"before":before,"after":after,"new_jobs":new,"seconds":elapsed})
        progress(str(evidence[-1]));return result
    data=record("Parquet schema loading","metadata","v2:metadata",lambda:spark.read.parquet(str(source/"analysis")))
    data=record("select()","transformation","v2:lazy",lambda:data.select("hhidpn","wave","age","age_band","raeduc","sleep_quality","recall"))
    data=record("filter()","transformation","v2:lazy",lambda:data.filter((F.col("wave")>=12)&F.col("raeduc").isNotNull()))
    data=record("withColumn()","transformation","v2:lazy",lambda:data.withColumn("recall_fraction",F.col("recall")/20))
    summary=record("groupBy().agg()","transformation","v2:lazy",lambda:data.groupBy("wave","age_band","raeduc","sleep_quality")
                   .agg(F.count("*").alias("observations"),F.avg("recall_fraction").alias("mean_recall_fraction")))
    record("explain()","transformation","v2:lazy",lambda:save_plans(summary,root,"before_actions"))
    count=record("count()","action","v2:count",lambda:data.count())
    def show():
        capture=io.StringIO()
        with contextlib.redirect_stdout(capture):summary.show(10,truncate=False)
        progress(capture.getvalue())
    record("show(10, truncate=False)","action","v2:show",show)
    rows=record("bounded collect for verification","action","v2:verify",lambda:[r.asDict() for r in summary.limit(4000).collect()])
    require(len(rows)<4000,"Aggregate output unexpectedly large.")
    expected=[{k:r[k] for k in ["wave","age_band","raeduc","sleep_quality","observations"]} | {"mean_recall_fraction":r.mean_recall/20}
              for r in spark.read.parquet(str(source/"sleep_recall_summary")).filter("wave>=12").collect()]
    rows_equal(rows,expected,["wave","age_band","raeduc","sleep_quality"])
    require(count==sum(r["observations"] for r in rows),"Stage 3 counts differ.")
    with (root/"summary_results.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(root/"operation_evidence.json",evidence)
    (root/"actions_vs_transformations.txt").write_text(
        "select, filter, withColumn and groupBy().agg define a plan; no jobs were submitted in their job group. "
        "Metadata loading is tracked separately. count and show submit jobs; a single action may launch several jobs. "
        "count can prune unused expressions; the displayed grouped mean consumes recall_fraction. "
        "Only bounded aggregates are collected for validation. Age bands, education and three sleep-quality levels define the comparisons.\n")
    manifest["counts"]={"qualifying_observations":count,"summary_groups":len(rows)}
    manifest["verification"]={"lazy_group_jobs":tracker.getJobIdsForGroup("v2:lazy"),"actions_triggered_jobs":True,"stage1_agreement":True}


def run_stage4(args,spark,root,manifest):
    """Stage 4 — adjusted associations, age interactions, and held-out prediction."""
    import numpy as np
    from pyspark import StorageLevel
    from pyspark.ml import Pipeline, PipelineModel
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
    from pyspark.ml.regression import LinearRegression
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.functions import vector_to_array
    source=read_analysis(args,manifest)
    data=spark.read.parquet(str(source/"analysis"))
    # Respondent-only split prevents a person's waves entering both sets.
    data=(data.withColumn("split_bucket",F.pmod(F.xxhash64("hhidpn",F.lit(42)),F.lit(10000)))
          .withColumn("split",F.when(F.col("split_bucket")<8000,"train").otherwise("test"))
          .persist(StorageLevel.MEMORY_AND_DISK))
    require(data.filter(~F.col("recall").between(0,20) | ~F.col("raeduc").isin(1,2,3,4,5) |
                        ~F.col("sleep_quality").isin("high","medium","low")).count()==0,"Invalid primary values.")
    train,test=data.filter("split='train'"),data.filter("split='test'")
    counts={}
    for name,frame in [("train",train),("test",test)]:
        counts[name]=frame.agg(F.count("*").alias("observations"),F.countDistinct("hhidpn").alias("respondents")).first().asDict()
        require(counts[name]["observations"]>0,"Empty split.")
    overlap=train.select("hhidpn").distinct().intersect(test.select("hhidpn").distinct()).count()
    require(overlap==0,"Respondent leakage.")
    mean=train.agg(F.avg("recall")).first()[0]
    manifest["counts"]=counts;manifest["split"]={"seed":42,"method":"pmod(xxhash64(hhidpn,42),10000)<8000","overlap":overlap}
    def features(frame):
        # High quality is the explicit reference; age is centered at 65.
        return (frame.withColumn("age_centered",F.col("age")-65.)
                .withColumn("sleep_medium",(F.col("sleep_quality")=="medium").cast("double"))
                .withColumn("sleep_low",(F.col("sleep_quality")=="low").cast("double"))
                .withColumn("age_x_medium",F.col("age_centered")*F.col("sleep_medium"))
                .withColumn("age_x_low",F.col("age_centered")*F.col("sleep_low")))
    train,test=features(train),features(test)
    for col in ("raeduc","wave","sleep_quality"):
        require(test.select(col).distinct().join(train.select(col).distinct(),col,"left_anti").count()==0,
                f"Unseen test categories in {col}.")
    def evaluate(frame,column):
        result={m:RegressionEvaluator(labelCol="recall",predictionCol=column,metricName=m).evaluate(frame) for m in ("rmse","mae","r2")}
        row=frame.agg(F.avg(F.pow(F.col(column)-F.col("recall"),2)).alias("mse"),
                      F.avg(F.abs(F.col(column)-F.col("recall"))).alias("mae"),F.var_pop("recall").alias("var")).first()
        require(row.var>0,"Zero test variance.")
        for metric,value in {"rmse":math.sqrt(row.mse),"mae":row.mae,"r2":1-row.mse/row.var}.items():
            require(math.isfinite(result[metric]) and math.isclose(value,result[metric],abs_tol=1e-8,rel_tol=1e-8),"Metric mismatch.")
        return result
    predictions=test.withColumn("baseline_prediction",F.lit(mean))
    metrics={"baseline":evaluate(predictions,"baseline_prediction")}
    models={};coefficient_records=[];references={};ranks={};reload_errors={}
    for name in ("additive","interaction"):
        cols=["age_centered","sleep_medium","sleep_low","raeduc_encoded","wave_encoded"]
        if name=="interaction":cols += ["age_x_medium","age_x_low"]
        indexers=[StringIndexer(inputCol=c,outputCol=c+"_index",stringOrderType="alphabetAsc",handleInvalid="error") for c in ("raeduc","wave")]
        encoder=OneHotEncoder(inputCols=["raeduc_index","wave_index"],outputCols=["raeduc_encoded","wave_encoded"],dropLast=True)
        pipeline=Pipeline(stages=[*indexers,encoder,VectorAssembler(inputCols=cols,outputCol="features",handleInvalid="error"),
            LinearRegression(featuresCol="features",labelCol="recall",regParam=0.,elasticNetParam=0.,solver="normal",standardization=True)])
        progress(f"Fitting {name} OLS on training respondents only: {counts}")
        model=pipeline.fit(train);models[name]=model
        predicted=model.transform(test).persist(StorageLevel.MEMORY_AND_DISK)
        require(predicted.filter(F.isnan("prediction") | (F.abs("prediction")==float('inf')) |
                                F.exists(vector_to_array("features"),lambda x:F.isnan(x)|(F.abs(x)==float('inf')))).count()==0,"Nonfinite vectors/predictions.")
        require(predicted.count()==counts["test"]["observations"],"Prediction row loss.")
        metrics[name]=evaluate(predicted,"prediction")
        metrics[name]["out_of_range_predictions"]=predicted.filter(~F.col("prediction").between(0,20)).count()
        attrs=predicted.schema["features"].metadata["ml_attr"]
        names=sorted([a for group in attrs["attrs"].values() for a in group],key=lambda a:a["idx"])
        beta=list(map(float,model.stages[-1].coefficients))
        require(len(names)==len(beta)==attrs["num_attrs"] and [a["idx"] for a in names]==list(range(len(beta))),"Feature index mismatch.")
        expected=["age_centered","sleep_medium","sleep_low"]
        for col,indexer in zip(("raeduc","wave"),model.stages[:2]):
            expected += [f"{col}_encoded_{label}" for label in indexer.labels[:-1]]
            references[col]=indexer.labels[-1]
        if name=="interaction":expected += ["age_x_medium","age_x_low"]
        require(expected==[a["name"] for a in names],"Coefficient labels differ.")
        require(all(math.isfinite(x) for x in beta+[model.stages[-1].intercept]),"Nonfinite coefficients.")
        coefficient_records += [{"model":name,"feature":a["name"],"coefficient":b} for a,b in zip(names,beta)]
        coefficient_records.append({"model":name,"feature":"intercept","coefficient":float(model.stages[-1].intercept)})
        # Small Gram matrix only: verify rank without collecting respondent rows.
        vectors=model.transform(train).select(vector_to_array("features").alias("v"))
        xs=[F.lit(1.)]+[F.col("v")[i] for i in range(len(beta))]
        gram_row=vectors.agg(*[F.sum(xs[i]*xs[j]).alias(f"g{i}_{j}") for i in range(len(xs)) for j in range(i,len(xs))]).first()
        gram=np.zeros((len(xs),len(xs)))
        for i in range(len(xs)):
            for j in range(i,len(xs)):gram[i,j]=gram[j,i]=gram_row[f"g{i}_{j}"]
        # Normalize columns before checking numerical rank.
        scale=np.sqrt(np.diag(gram));normalized=gram/np.outer(scale,scale)
        rank=int(np.linalg.matrix_rank(normalized))
        require(rank==len(xs),f"Rank-deficient {name} design.")
        ranks[name]={"rank":rank,"columns_including_intercept":len(xs),"scaled_gram_condition":float(np.linalg.cond(normalized))}
        model.write().save(str(root/(name+"_model")))
        loaded=PipelineModel.load(str(root/(name+"_model")))
        again=loaded.transform(test).select("hhidpn","wave",F.col("prediction").alias("reloaded"))
        joined=predicted.select("hhidpn","wave","prediction").join(again,["hhidpn","wave"],"full")
        difference=joined.agg(F.max(F.abs(F.col("prediction")-F.col("reloaded")))).first()[0]
        require(difference is not None and difference<1e-10 and joined.filter("prediction IS NULL OR reloaded IS NULL").count()==0,"Model reload mismatch.")
        reload_errors[name]=difference
        predictions=predictions.join(predicted.select("hhidpn","wave",F.col("prediction").alias(name+"_prediction")),["hhidpn","wave"])
        predicted.unpersist()
        progress(f"{name} test metrics: {metrics[name]}")
    with (root/"coefficients.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["model","feature","coefficient"]);writer.writeheader();writer.writerows(coefficient_records)
    grid=spark.createDataFrame([(a,q,3,12) for a in (55.,65.,75.,85.) for q in ("high","medium","low")],
                               ["age","sleep_quality","raeduc","wave"])
    adjusted=[];contrasts=[]
    beta={r["feature"]:r["coefficient"] for r in coefficient_records if r["model"]=="interaction"}
    for name,model in models.items():
        rows=[r.asDict() for r in model.transform(features(grid)).select("age","sleep_quality","raeduc","wave","prediction").collect()]
        lookup={(r["age"],r["sleep_quality"]):r["prediction"] for r in rows}
        for r in rows:
            support=train.filter((F.col("raeduc")==3)&(F.col("wave")==12)&(F.col("sleep_quality")==r["sleep_quality"])&
                                 F.col("age").between(r["age"]-5,r["age"]+5)).count()
            adjusted.append({"model":name,**r,"training_support_within_5_years":support,"supported_at_least_20":support>=20})
        for age in (55.,65.,75.,85.):
            for quality in ("medium","low"):
                value=lookup[(age,quality)]-lookup[(age,"high")]
                if name=="interaction":
                    require(math.isclose(value,beta["sleep_"+quality]+beta["age_x_"+quality]*(age-65),abs_tol=1e-9),"Interaction contrast mismatch.")
                contrasts.append({"model":name,"age":age,"contrast":quality+"-high","difference":value})
    write_json(root/"adjusted_predictions.json",adjusted);write_json(root/"adjusted_contrasts.json",contrasts)
    for name,rows in [("adjusted_predictions",adjusted),("adjusted_contrasts",contrasts)]:
        with (root/(name+".csv")).open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    require(save_parquet(predictions.select("hhidpn","wave","age","raeduc","sleep_quality","recall","baseline_prediction","additive_prediction","interaction_prediction"),root,"test_predictions")==counts["test"]["observations"],"Prediction output mismatch.")
    require(save_parquet(data.select("hhidpn","split_bucket","split").distinct(),root,"respondent_splits")==sum(counts[n]["respondents"] for n in counts),"Split output mismatch.")
    manifest.update(metrics=metrics,reference_categories={"sleep_quality":"high",**references},design_checks=ranks,
                    adjusted_profile={"education":3,"wave":12,"ages":[55,65,75,85]},baseline_training_mean=mean)
    write_json(root/"metrics.json",metrics)
    (root/"model_summary.txt").write_text(
        f"Primary cohort, respondent-disjoint split: {counts}. Metrics: {metrics}. Additive and interaction OLS use identical training/testing respondents. "
        "Sleep reference is high; age is centered at 65; education/wave are categorical. Sleep coefficients are differences at 65 in the interaction model. "
        "Age-specific comparisons hold education at high-school graduate and wave at 12; support is reported. "
        "These are pooled adjusted associations and predictions, not causal or within-person aging effects. "
        "No respondent-clustered inferential test was fitted; no p-values, confidence intervals or significance claims are made. "
        "Complete-case selection, attrition, arbitrary sleep cutoffs, equal item weights, unweighted repeated observations and household dependence limit interpretation.\n")
    manifest["verification"]={"zero_overlap":True,"finite_features_predictions":True,"full_rank":True,"feature_alignment":True,
        "metrics_reconciled":True,"model_reload_max_errors":reload_errors,"interaction_contrasts_checked":True}
    data.unpersist()


def main():
    global ACTIVE_STAGE,ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",type=int,choices=(1,2,3,4),required=True)
    parser.add_argument("--input",default="randhrs1992_2022v2.dta")
    parser.add_argument("--output-dir",default=None)
    parser.add_argument("--stage1-dir",default="output/research_v2/stage1")
    parser.add_argument("--chunk-size",type=int,default=5000)
    parser.add_argument("--master",default="local[2]")
    parser.add_argument("--shuffle-partitions",type=int,default=4)
    parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--overwrite",action="store_true",help="Replace only an owned revised stage directory")
    args=parser.parse_args()
    if min(args.chunk_size,args.shuffle_partitions,args.repeats)<1:parser.error("Counts must be positive.")
    if not (args.master=="local" or (args.master.startswith("local[") and args.master.endswith("]"))):parser.error("Use local Spark.")
    root=Path(args.output_dir or f"output/research_v2/stage{args.stage}").resolve()
    revision_root=Path("output/research_v2").resolve()
    if root.parent!=revision_root or root.name!=f"stage{args.stage}":parser.error("Use output/research_v2/stageN; old outputs are protected.")
    if root.exists():
        if not args.overwrite:parser.error(f"Output exists: {root}; use --overwrite intentionally.")
        marker=root/"run_manifest.json"
        if not marker.is_file() or json.loads(marker.read_text()).get("owner")!=f"research_v2_stage{args.stage}":parser.error("Output is not owned by revised stage.")
        require(not root.is_symlink() and not any(p.is_symlink() for p in root.rglob('*')),"Refusing symlink overwrite.")
        shutil.rmtree(root)
    root.mkdir(parents=True);ACTIVE_STAGE=args.stage;ROOT=root
    started=time.perf_counter();spark=None
    manifest={"owner":f"research_v2_stage{args.stage}","schema_version":SCHEMA_VERSION,"status":"incomplete",
              "arguments":vars(args),"started_utc":datetime.now(timezone.utc).isoformat(),
              "versions":{p:version(p) for p in ("pyspark","pyarrow","pyreadstat","numpy")}}
    write_json(root/"run_manifest.json",manifest)
    try:
        os.environ.setdefault("SPARK_LOCAL_IP","127.0.0.1")
        os.environ["PYSPARK_PYTHON"]=sys.executable
        builder=(SparkSession.builder.master(args.master).appName(f"IDS706 Research V2 Stage {args.stage}")
                 .config("spark.ui.enabled",str(args.stage==2).lower()).config("spark.ui.showConsoleProgress","false")
                 .config("spark.pyspark.python",sys.executable)
                 .config("spark.sql.shuffle.partitions",args.shuffle_partitions).config("spark.sql.adaptive.enabled","true")
                 .config("spark.sql.session.timeZone","UTC").config("spark.sql.warehouse.dir",(root/"warehouse").as_uri())
                 .config("spark.local.dir",str(root/"spark_local")).config("spark.sql.artifact.isolation.enabled","false"))
        if args.stage in (2,3):
            (root/"event_logs").mkdir();builder=builder.config("spark.eventLog.enabled","true").config("spark.eventLog.dir",(root/"event_logs").as_uri())
        spark=builder.getOrCreate();spark.sparkContext.setLogLevel("ERROR")
        manifest["jvm_max_heap_bytes"]=spark._jvm.java.lang.Runtime.getRuntime().maxMemory()
        {1:run_stage1,2:run_stage2,3:run_stage3,4:run_stage4}[args.stage](args,spark,root,manifest)
        manifest["status"]="complete"
    except Exception as error:
        manifest.update(status="failed",error=f"{type(error).__name__}: {error}")
        progress(manifest["error"])
    finally:
        if spark is not None:
            try:spark.stop()
            except Exception as error:manifest["cleanup_error"]=str(error)
        manifest["elapsed_seconds"]=round(time.perf_counter()-started,2)
        write_json(root/"run_manifest.json",manifest)
        progress(f"Status {manifest['status']}; elapsed {manifest['elapsed_seconds']}s; output {root}")
    return 0 if manifest["status"]=="complete" else 1


if __name__=="__main__":
    sys.exit(main())
