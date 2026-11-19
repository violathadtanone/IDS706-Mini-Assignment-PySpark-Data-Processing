"""IDS706: PySpark data processing, performance, lazy evaluation, and ML.

Run a stage: .venv/bin/python -B pyspark_pipeline.py --stage {1,2,3,4}
RAND HRS waves 6–16 (2002–2022); results are predictive or descriptive, not causal.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version


CODEBOOK = ("https://hrsdata.isr.umich.edu/sites/default/files/documentation/"
            "other/1793653601/randhrs1992_2022v2.pdf")
OWNER = "ids706-pyspark-stage1-v1"
WAVES = range(6, 17)
SLEEP = ("sleepfal", "sleepwkn", "sleepwke", "sleeprt")
PRODUCTS = ("input_selected", "respondent_wave", "wave_summary",
            "respondent_summary", "longitudinal_summary", "sleep_recall_summary",
            "spark_local", "warehouse")


def progress(message):
    print(f"[Stage 1] {message}", flush=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(root, manifest):
    temporary = root / "run_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    temporary.replace(root / "run_manifest.json")


def prepare_output(root, source, overwrite):
    require(root != Path(root.anchor) and root != Path.cwd().resolve(),
            "Choose a dedicated Stage 1 output directory.")
    protected = [source, Path("randhrs.parquet").resolve(), Path(__file__).resolve()]
    require(not any(p == root or root in p.parents for p in protected),
            "Output must not contain source data or this script.")
    require(not root.is_symlink(), "Output directory cannot be a symlink.")
    if root.exists():
        require(overwrite, f"Output exists: {root}. Use --overwrite intentionally.")
        marker = root / "run_manifest.json"
        require(marker.is_file(), "Refusing overwrite: no Stage 1 ownership manifest.")
        require(json.loads(marker.read_text()).get("owner") == OWNER,
                "Refusing overwrite: directory does not belong to this script.")
        allowed = set(PRODUCTS) | {"run_manifest.json", "run_manifest.json.tmp"}
        require(all(p.name in allowed and not p.is_symlink() for p in root.iterdir()),
                "Refusing overwrite: unexpected files or symlinks in output.")
        for name in PRODUCTS:
            path = root / name
            if path.exists():
                require(path.is_dir(), f"Expected an output directory: {path}")
                shutil.rmtree(path)
    else:
        root.mkdir(parents=True)


def convert_selected(source, root, chunk_size):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyreadstat

    _, metadata = pyreadstat.read_dta(source, metadataonly=True)
    columns = ["hhid", "pn", "hhidpn"]
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


def run_stage1(args):
    """Stage 1 — Data Processing Pipeline."""
    from pyspark.sql import SparkSession, functions as F
    from pyspark import StorageLevel

    started = time.perf_counter()
    source = Path(args.input).expanduser().resolve()
    raw_output = Path(args.output_dir).expanduser().absolute()
    require(not raw_output.is_symlink(), "Output directory cannot be a symlink.")
    root = raw_output.resolve()
    require(source.is_file() and source.suffix.lower() == ".dta", "Input must be an existing Stata .dta file.")
    progress("Checking source identity and size")
    source_stat = source.stat()
    fingerprint = sha256(source)
    prepare_output(root, source, args.overwrite)
    manifest = {
        "owner": OWNER, "status": "incomplete", "started_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args), "source": str(source), "source_bytes": source_stat.st_size,
        "source_sha256": fingerprint, "source_at_least_1GB": source_stat.st_size >= 10**9,
        "versions": {p: version(p) for p in ("pandas", "pyreadstat", "pyarrow", "pyspark")},
        "documentation": CODEBOOK,
        "rules": {
            "waves": list(WAVES), "age": "integer age >=50, analytical cohort restriction",
            "interview_status": [1, 2], "proxy": "retain in panel; exclude from joint analysis",
            "sleep": "1=most, 2=sometimes, 3=rarely/never; other values null",
            "frequent_sleep_trouble": "all four valid; any trouble item=1 or rested=3",
            "recall": "0..20 integer; face-to-face/telephone and web kept separate",
            "missing": "Stata system/extended missing -> null; no carry-forward or zero filling",
            "inference": "unweighted descriptive associations; not causal or population estimates",
        },
    }
    write_manifest(root, manifest)
    spark = None
    panel = None
    try:
        progress(f"Source {source_stat.st_size:,} bytes ({source_stat.st_size / 10**9:.3f} GB)")
        progress("Converting selected columns in bounded chunks")
        manifest.update(convert_selected(source, root, args.chunk_size))
        write_manifest(root, manifest)
        progress(f"Input dimensions: {manifest['source_rows']:,} x {manifest['source_columns']:,}; "
                 f"selected columns: {manifest['selected_schema_columns']}")

        os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
        spark = (SparkSession.builder.master(args.master).appName("IDS706 Stage 1")
                 .config("spark.sql.shuffle.partitions", args.shuffle_partitions)
                 .config("spark.sql.adaptive.enabled", "true")
                 .config("spark.sql.parquet.mergeSchema", "false")
                 .config("spark.sql.session.timeZone", "UTC")
                 .config("spark.sql.warehouse.dir", (root / "warehouse").as_uri())
                 .config("spark.local.dir", str(root / "spark_local"))
                 .config("spark.ui.enabled", "false")
                 .config("spark.sql.artifact.isolation.enabled", "false")
                 .getOrCreate())
        spark.sparkContext.setLogLevel("ERROR")
        manifest["spark"] = {
            "version": spark.version, "master": spark.sparkContext.master,
            "java_version": spark._jvm.java.lang.System.getProperty("java.version"),
            "jvm_max_heap_bytes": spark._jvm.java.lang.Runtime.getRuntime().maxMemory(),
            "shuffle_partitions": args.shuffle_partitions,
        }
        progress("Loading data into PySpark")
        # Optimize: select only the required columns for processing.
        wide = spark.read.parquet(str(root / "input_selected")).select(*manifest["selected_columns"])
        ids = wide.agg(F.count("*").alias("rows"),
                       F.countDistinct("hhidpn").alias("ids"),
                       F.sum(F.when(~F.col("hhid").rlike(r"^\d{6}$") |
                                    ~F.col("pn").rlike(r"^\d{3}$") |
                                    F.col("hhid").isNull() | F.col("pn").isNull() |
                                    (F.concat("hhid", "pn").cast("long") != F.col("hhidpn")), 1)
                             .otherwise(0)).alias("invalid_ids")).first().asDict()
        require(ids["rows"] == ids["ids"] == manifest["source_rows"] and ids["invalid_ids"] == 0,
                f"Identifier or conversion validation failed: {ids}")
        manifest["identifier_check"] = ids

        records = []
        for wave in WAVES:
            def variable(suffix):
                return F.col(f"r{wave}{suffix}")

            fields = [F.lit(wave).alias("wave"), F.lit(1990 + 2 * wave).alias("survey_year"),
                      variable("iwendy").alias("interview_year"), variable("agey_b").alias("age"),
                      variable("iwstat").alias("interview_status"), variable("proxy").alias("proxy")]
            fields.extend(variable(s).alias(s) for s in SLEEP)
            for mode, suffix in (("ftf_tel", "" if wave < 14 else "p"), ("web", "w")):
                for name in ("tr20", "fimrc", "fdlrc"):
                    value = variable(name + suffix) if mode == "ftf_tel" or wave >= 14 else F.lit(None).cast("short")
                    fields.append(value.alias(f"{name}_{mode}"))
            records.append(F.struct(*fields))
        long = wide.select("hhid", "pn", "hhidpn", F.explode(F.array(*records)).alias("observation"))
        long = long.select("hhid", "pn", "hhidpn", "observation.*")
        progress("Applying filters and column transformations")
        # Filter: restrict interviews and age before aggregation.
        participating = long.filter(F.col("interview_status").isin(1, 2))
        cohort = participating.filter(F.col("age") >= 50)
        # Transform: retain valid codes; leave missing values null.
        for name in SLEEP:
            cohort = cohort.withColumn(name, F.when(F.col(name).isin(1, 2, 3), F.col(name)))
        for mode in ("ftf_tel", "web"):
            name = f"tr20_{mode}"
            cohort = cohort.withColumn(name, F.when(F.col(name).between(0, 20), F.col(name)))
            for flag in ("fimrc", "fdlrc"):
                name = f"{flag}_{mode}"
                cohort = cohort.withColumn(name, F.when(F.col(name).isin(0, 1, 2), F.col(name)))
        complete_sleep = F.lit(True)
        for name in SLEEP:
            complete_sleep = complete_sleep & F.col(name).isNotNull()
        frequent = ((F.col("sleepfal") == 1) | (F.col("sleepwkn") == 1) |
                    (F.col("sleepwke") == 1) | (F.col("sleeprt") == 3))
        panel = (cohort.withColumn("frequent_sleep_trouble", F.when(complete_sleep, frequent.cast("int")))
                 .withColumn("age_band", F.when(F.col("age") < 65, "50-64")
                             .when(F.col("age") < 75, "65-74")
                             .when(F.col("age") < 85, "75-84").otherwise("85+"))
                 .persist(StorageLevel.MEMORY_AND_DISK))
        # Counts below are respondent-waves, not unique respondents.
        counts = {"potential_respondent_waves": manifest["source_rows"] * len(WAVES),
                  "participating_respondent_waves": participating.count(),
                  "age50_respondent_waves": panel.count()}
        require(panel.groupBy("hhidpn", "wave").count().filter("count > 1").limit(1).count() == 0,
                "Duplicate respondent-wave keys.")
        modes = F.array(*[F.struct(F.lit(mode).alias("cognition_mode"),
                                   F.col(f"tr20_{mode}").alias("recall"),
                                   F.col(f"fimrc_{mode}").alias("immediate_imputation_flag"),
                                   F.col(f"fdlrc_{mode}").alias("delayed_imputation_flag"))
                           for mode in ("ftf_tel", "web")])
        measured = (panel.select("*", F.explode(modes).alias("measurement"))
                    .select("hhidpn", "wave", "age", "age_band", "frequent_sleep_trouble", "proxy", "measurement.*")
                    .filter(F.col("recall").isNotNull()))
        joint = measured.filter((F.col("proxy") == 0) & F.col("frequent_sleep_trouble").isNotNull())
        counts["joint_complete_mode_observations"] = joint.count()
        manifest["counts"] = counts
        progress(f"Filtered row counts: {counts}")

        # Aggregate: groupBy with counts and means.
        wave_summary = panel.groupBy("wave", "survey_year", "age_band").agg(
            F.count("*").alias("respondent_waves"), F.avg("age").alias("mean_age"),
            F.count("frequent_sleep_trouble").alias("complete_sleep_n"),
            F.avg("frequent_sleep_trouble").alias("frequent_sleep_proportion"),
            F.count("tr20_ftf_tel").alias("ftf_tel_recall_n"),
            F.avg("tr20_ftf_tel").alias("ftf_tel_mean_recall"),
            F.count("tr20_web").alias("web_recall_n"),
            F.avg("tr20_web").alias("web_mean_recall"))
        joint.createOrReplaceTempView("joint_observations")
        progress("Running Spark SQL query 1: sleep and recall by wave, age, and mode")
        # SQL 1: recall by wave, age, sleep, and interview mode.
        sql_summary = spark.sql("""
            SELECT wave, age_band, cognition_mode, frequent_sleep_trouble,
                   COUNT(*) AS observations, COUNT(DISTINCT hhidpn) AS respondents,
                   AVG(recall) AS mean_recall, STDDEV_SAMP(recall) AS sd_recall,
                   SUM(CASE WHEN immediate_imputation_flag=1 OR delayed_imputation_flag=1
                            THEN 1 ELSE 0 END) AS any_imputed_n
            FROM joint_observations
            GROUP BY wave, age_band, cognition_mode, frequent_sleep_trouble
        """)
        sql_summary.orderBy("wave", "age_band", "cognition_mode", "frequent_sleep_trouble").show(12, truncate=False)
        progress("Running Spark SQL query 2: within-mode longitudinal change")
        # SQL 2: longitudinal change using CTEs and first/last observations.
        spark.sql("""
            CREATE OR REPLACE TEMP VIEW respondent_changes AS
            WITH repeated AS (
                SELECT hhidpn, cognition_mode, COUNT(*) AS observed_waves,
                       AVG(frequent_sleep_trouble) AS sleep_trouble_proportion,
                       MIN(named_struct('wave',wave,'recall',recall)) AS first_observation,
                       MAX(named_struct('wave',wave,'recall',recall)) AS last_observation
                FROM joint_observations GROUP BY hhidpn, cognition_mode HAVING COUNT(*)>=2
            )
            SELECT hhidpn, cognition_mode, observed_waves, sleep_trouble_proportion,
                   first_observation.wave AS first_wave, last_observation.wave AS last_wave,
                   first_observation.recall AS first_recall, last_observation.recall AS last_recall,
                   last_observation.recall-first_observation.recall AS recall_change,
                   (last_observation.wave-first_observation.wave)*2 AS nominal_elapsed_years,
                   CASE WHEN sleep_trouble_proportion>=0.5 THEN 'at_least_half'
                        ELSE 'less_than_half' END AS sleep_pattern
            FROM repeated
        """)
        longitudinal = spark.sql("""
            SELECT cognition_mode, sleep_pattern, COUNT(*) AS respondents,
                   AVG(observed_waves) AS mean_observed_waves,
                   AVG(nominal_elapsed_years) AS mean_nominal_elapsed_years,
                   AVG(recall_change) AS mean_recall_change,
                   STDDEV_SAMP(recall_change) AS sd_recall_change
            FROM respondent_changes GROUP BY cognition_mode, sleep_pattern
        """)
        longitudinal.orderBy("cognition_mode", "sleep_pattern").show(truncate=False)
        respondent_summary = spark.table("respondent_changes")

        group = ["wave", "age_band", "cognition_mode", "frequent_sleep_trouble"]
        expected = joint.groupBy(*group).agg(F.count("*").alias("n"), F.avg("recall").alias("avg"))
        mismatches = (expected.join(sql_summary, group, "full")
                      .filter(F.col("n").isNull() | F.col("observations").isNull() |
                              (F.col("n") != F.col("observations")) |
                              (F.abs(F.col("avg") - F.col("mean_recall")) > 1e-10)).count())
        require(mismatches == 0, "SQL/DataFrame aggregation mismatch.")
        require(wave_summary.agg(F.sum("respondent_waves")).first()[0] == counts["age50_respondent_waves"],
                "Wave summary denominators do not reconcile.")

        progress("Writing processed Parquet results")
        # Write: processed results to Parquet.
        # Coalesce limits output files without another shuffle.
        outputs = {"respondent_wave": panel, "wave_summary": wave_summary,
                   "respondent_summary": respondent_summary,
                   "longitudinal_summary": longitudinal}
        manifest["outputs"] = {}
        for name, frame in outputs.items():
            path = root / name
            frame.coalesce(args.shuffle_partitions if name == "respondent_wave" else 1).write.mode("errorifexists").parquet(str(path))
            reread = spark.read.parquet(str(path))
            output_rows = reread.count()
            require([(f.name, f.dataType) for f in reread.schema] ==
                    [(f.name, f.dataType) for f in frame.schema], f"Read-back schema mismatch: {name}")
            if name == "respondent_wave":
                require(output_rows == counts["age50_respondent_waves"], "Panel read-back count mismatch.")
            manifest["outputs"][name] = {"path": str(path), "rows": output_rows,
                "parquet_files": len(list(path.glob("*.parquet")))}
            progress(f"{name}: {output_rows:,} rows -> {path}")
        sql_path = root / "sleep_recall_summary"
        sql_summary.coalesce(1).write.parquet(str(sql_path))
        manifest["sql_summary_rows"] = spark.read.parquet(str(sql_path)).count()
        manifest["imputation_flags"] = [r.asDict() for r in measured.groupBy(
            "cognition_mode", "immediate_imputation_flag", "delayed_imputation_flag").count().collect()]
        manifest["verification"] = {"sql_dataframe_mismatches": mismatches,
                                    "unique_respondent_wave_keys": True,
                                    "source_sha256_unchanged": sha256(source) == fingerprint}
        require(manifest["verification"]["source_sha256_unchanged"], "Source checksum changed.")
        manifest["status"] = "complete"
        progress("Verification passed; results are unweighted descriptive associations")
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if spark is not None:
            try:
                if panel is not None:
                    panel.unpersist()
                spark.stop()
            except Exception as error:
                manifest["cleanup_error"] = str(error)
        manifest["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        write_manifest(root, manifest)
        progress(f"Status: {manifest['status']}; elapsed {manifest['elapsed_seconds']:.2f}s; manifest: {root / 'run_manifest.json'}")


def run_stage2(args):
    """Stage 2 — Performance Analysis."""
    import contextlib
    import csv
    import io
    import math
    import statistics
    import struct
    import subprocess
    import urllib.request
    import zlib
    from pyspark.sql import SparkSession, functions as F

    def log(message):
        print(f"[Stage 2] {message}", flush=True)

    source = Path(args.stage1_dir).expanduser().resolve()
    raw_root = Path(args.output_dir).expanduser().absolute()
    root = raw_root.resolve()
    require(not raw_root.is_symlink(), "Output directory cannot be a symlink.")
    require(root != source and root not in source.parents and source not in root.parents,
            "Stage 2 output must be separate from Stage 1.")
    require(root != Path.cwd().resolve() and root not in Path(__file__).resolve().parents,
            "Choose a dedicated Stage 2 output directory.")
    stage1 = json.loads((source / "run_manifest.json").read_text())
    require(stage1.get("status") == "complete", "Stage 1 must have completed successfully.")
    marker = root / "run_manifest.json"
    owner = "ids706-pyspark-stage2-v1"
    if root.exists():
        require(args.overwrite, f"Output exists: {root}. Use --overwrite intentionally.")
        require(marker.is_file() and json.loads(marker.read_text()).get("owner") == owner,
                "Refusing overwrite: no Stage 2 ownership manifest.")
        allowed = {"plans", "screenshots", "event_logs", "spark_local", "warehouse",
                   "chrome_profile", "ui_evidence", "timings.csv", "performance_analysis.txt",
                   "run_manifest.json", "run_manifest.json.tmp"}
        require(all(p.name in allowed and not p.is_symlink() for p in root.iterdir()),
                "Refusing overwrite: unexpected files or symlinks.")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for name in ("plans", "screenshots", "event_logs", "ui_evidence"):
        (root / name).mkdir()
    protected = [p for p in source.rglob("*") if p.is_file()]
    protected += [Path(n) for n in ("README.md", "requirements.txt", ".gitignore") if Path(n).is_file()]
    before = {str(p): sha256(p) for p in protected}
    started = time.perf_counter()
    manifest = {"owner": owner, "status": "incomplete", "arguments": vars(args),
                "started_utc": datetime.now(timezone.utc).isoformat(), "stage1_dir": str(source),
                "protected_checksums_before": before, "screenshots": [],
                "timing_scope": "collect small aggregate results; query construction, explain, screenshots and writes excluded",
                "benchmark_caveats": "No Spark cache in uncached trials; OS cache and JVM warm-up remain. Fixed condition order may bias timings."}
    write_manifest(root, manifest)
    spark = None
    cached = None
    timings = []

    def same_results(left, right):
        left = sorted([dict(r) for r in left], key=lambda r: str(sorted(r.items())))
        right = sorted([dict(r) for r in right], key=lambda r: str(sorted(r.items())))
        require(len(left) == len(right), "Aggregate result row counts differ.")
        # Match rows by categorical keys, avoiding floating-point sort differences.
        def key(row):
            return tuple(str(row[k]) for k in sorted(row) if k in
                         ("wave", "age_band", "frequent_sleep_trouble", "cognition_mode", "sleep_pattern"))
        for a, b in zip(sorted(left, key=key), sorted(right, key=key)):
            require(a.keys() == b.keys(), "Aggregate result schemas differ.")
            for k in a:
                x, y = a[k], b[k]
                equal = math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9) if isinstance(x, (int, float)) and isinstance(y, (int, float)) else x == y
                require(equal, f"Aggregate mismatch for {k}: {x} != {y}")

    def save_plan(frame, label, phase):
        for mode in ("formatted", "extended"):
            capture = io.StringIO()
            # Explain: display and save execution plans.
            with contextlib.redirect_stdout(capture):
                frame.explain(mode=mode)
            content = capture.getvalue()
            (root / "plans" / f"{label}_{phase}_{mode}.txt").write_text(content)
            if mode == "formatted":
                log(f"Physical plan: {label} ({phase})")
                print(content, flush=True)
        return content

    try:
        os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
        spark = (SparkSession.builder.master(args.master).appName("IDS706 Stage 2 Performance")
                 .config("spark.ui.enabled", "true")
                 .config("spark.ui.showConsoleProgress", "false")
                 .config("spark.eventLog.enabled", "true")
                 .config("spark.eventLog.dir", (root / "event_logs").as_uri())
                 .config("spark.sql.adaptive.enabled", "true")
                 .config("spark.sql.shuffle.partitions", 4)
                 .config("spark.sql.session.timeZone", "UTC")
                 .config("spark.sql.warehouse.dir", (root / "warehouse").as_uri())
                 .config("spark.local.dir", str(root / "spark_local"))
                 .config("spark.sql.artifact.isolation.enabled", "false").getOrCreate())
        spark.sparkContext.setLogLevel("ERROR")
        sc = spark.sparkContext
        ui = sc.uiWebUrl
        require(ui, "Spark UI did not start.")
        manifest["spark"] = {"version": spark.version, "application_id": sc.applicationId,
                             "ui_url": ui, "master": sc.master,
                             "jvm_max_heap_bytes": spark._jvm.java.lang.Runtime.getRuntime().maxMemory(),
                             "adaptive_enabled": spark.conf.get("spark.sql.adaptive.enabled")}
        log(f"Live Spark UI: {ui}")
        write_manifest(root, manifest)
        panel = spark.read.parquet(str(source / "respondent_wave"))
        needed = ["hhidpn", "wave", "age_band", "proxy", "frequent_sleep_trouble", "tr20_ftf_tel", "tr20_web"]
        require(set(needed) <= set(panel.columns), "Stage 1 panel lacks required columns.")
        require(panel.count() == stage1["counts"]["age50_respondent_waves"], "Stage 1 row count mismatch.")
        base = panel.select(*needed).filter((F.col("proxy") == 0) & F.col("frequent_sleep_trouble").isNotNull())
        manifest["input"] = {"rows": stage1["counts"]["age50_respondent_waves"],
                             "files": len(list((source / "respondent_wave").glob("*.parquet"))),
                             "bytes": sum(p.stat().st_size for p in (source / "respondent_wave").glob("*.parquet")),
                             "projected_columns": needed}

        def workload(name, data):
            if name == "filtered_recall":
                return (data.filter((F.col("wave") >= 12) & F.col("tr20_ftf_tel").isNotNull())
                        .groupBy("wave", "age_band", "frequent_sleep_trouble")
                        .agg(F.count("*").alias("observations"), F.countDistinct("hhidpn").alias("respondents"),
                             F.avg("tr20_ftf_tel").alias("mean_recall"), F.stddev_samp("tr20_ftf_tel").alias("sd_recall")))
            modes = F.array(*[F.struct(F.lit(mode).alias("cognition_mode"),
                                       F.col(f"tr20_{mode}").alias("recall")) for mode in ("ftf_tel", "web")])
            joint = (data.select("hhidpn", "wave", "frequent_sleep_trouble", F.explode(modes).alias("m"))
                     .select("hhidpn", "wave", "frequent_sleep_trouble", "m.*").filter("recall IS NOT NULL"))
            repeated = (joint.groupBy("hhidpn", "cognition_mode")
                        .agg(F.count("*").alias("observed_waves"),
                             F.avg("frequent_sleep_trouble").alias("sleep_trouble_proportion"),
                             F.min(F.struct("wave", "recall")).alias("first"),
                             F.max(F.struct("wave", "recall")).alias("last"))
                        .filter("observed_waves >= 2")
                        .withColumn("sleep_pattern", F.when(F.col("sleep_trouble_proportion") >= .5, "at_least_half").otherwise("less_than_half"))
                        .withColumn("recall_change", F.col("last.recall") - F.col("first.recall"))
                        .withColumn("nominal_elapsed_years", (F.col("last.wave") - F.col("first.wave")) * 2))
            return repeated.groupBy("cognition_mode", "sleep_pattern").agg(
                F.count("*").alias("respondents"), F.avg("observed_waves").alias("mean_observed_waves"),
                F.avg("nominal_elapsed_years").alias("mean_nominal_elapsed_years"),
                F.avg("recall_change").alias("mean_recall_change"), F.stddev_samp("recall_change").alias("sd_recall_change"))

        references = {
            "filtered_recall": [r.asDict() for r in spark.read.parquet(str(source / "sleep_recall_summary"))
                .filter("wave >= 12 AND cognition_mode = 'ftf_tel'")
                .select("wave", "age_band", "frequent_sleep_trouble", "observations", "respondents", "mean_recall", "sd_recall").collect()],
            "longitudinal_recall": [r.asDict() for r in spark.read.parquet(str(source / "longitudinal_summary")).collect()],
        }

        def trial(name, data, condition, partitions, phase, iteration, explain=False):
            spark.conf.set("spark.sql.shuffle.partitions", partitions)
            label = f"{name}_{condition}_p{partitions}"
            frame = workload(name, data)  # Fresh query prevents reusing an earlier AQE query object.
            if explain:
                save_plan(frame, label, "before")
            group = f"stage2:{label}:{phase}:{iteration}"
            sc.setJobGroup(group, group)
            start = time.perf_counter()
            result = [r.asDict() for r in frame.collect()]
            elapsed = time.perf_counter() - start
            jobs = list(sc.statusTracker().getJobIdsForGroup(group))
            sc.setLocalProperty("spark.jobGroup.id", None)
            sc.setLocalProperty("spark.job.description", None)
            same_results(result, references[name])
            timings.append({"workload": name, "condition": condition, "shuffle_partitions": partitions,
                            "phase": phase, "iteration": iteration, "seconds": elapsed,
                            "result_rows": len(result), "job_ids": json.dumps(jobs)})
            if explain:
                save_plan(frame, label, "after")
            log(f"{label} {phase} {iteration}: {elapsed:.4f}s ({len(result)} aggregate rows)")

        # Benchmark: warm up, repeat actions, and compare shuffle partitions.
        for partitions in (4, 2):
            for name in references:
                if partitions == 4:
                    trial(name, base, "uncached", partitions, "first", 0, explain=True)
                trial(name, base, "uncached", partitions, "warmup", 0)
                for i in range(1, args.repeats + 1):
                    trial(name, base, "uncached", partitions, "measured", i,
                          explain=(partitions == 2 and i == 1))

        spark.conf.set("spark.sql.shuffle.partitions", 4)
        # Cache bonus: persist the shared input, not final results.
        cached = base.cache()
        sc.setJobGroup("stage2:cache_population", "Stage 2 cache population (excluded from cached action timings)")
        start = time.perf_counter()
        cached_rows = cached.count()
        population = time.perf_counter() - start
        sc.setLocalProperty("spark.jobGroup.id", None)
        sc.setLocalProperty("spark.job.description", None)
        manifest["cache"] = {"population_seconds": population, "rows": cached_rows,
                             "storage_level": str(cached.storageLevel)}
        timings.append({"workload": "shared_input", "condition": "cached", "shuffle_partitions": 4,
                        "phase": "population", "iteration": 0, "seconds": population,
                        "result_rows": cached_rows, "job_ids": "[]"})
        for name in references:
            trial(name, cached, "cached", 4, "warmup", 0, explain=True)
            for i in range(1, args.repeats + 1):
                trial(name, cached, "cached", 4, "measured", i)

        with (root / "timings.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(timings[0]))
            writer.writeheader()
            writer.writerows(timings)
        summaries = []
        for name in references:
            for condition, partitions in (("uncached", 4), ("uncached", 2), ("cached", 4)):
                values = [t["seconds"] for t in timings if t["workload"] == name and t["condition"] == condition
                          and t["shuffle_partitions"] == partitions and t["phase"] == "measured"]
                summaries.append({"workload": name, "condition": condition, "shuffle_partitions": partitions,
                                  "median_seconds": statistics.median(values), "min_seconds": min(values),
                                  "max_seconds": max(values), "stdev_seconds": statistics.stdev(values) if len(values)>1 else 0,
                                  "total_seconds": sum(values), "trials": len(values)})
        manifest["timing_summaries"] = summaries

        api = f"{ui}/api/v1/applications/{sc.applicationId}"
        def fetch(url):
            with urllib.request.urlopen(url, timeout=30) as response:
                return response.read()

        snapshots = {}
        for endpoint in ("jobs", "stages", "sql", "storage/rdd", "executors"):
            payload = fetch(f"{api}/{endpoint}")
            (root / "ui_evidence" / f"{endpoint.replace('/', '_')}.json").write_bytes(payload)
            snapshots[endpoint] = json.loads(payload)
        stages = [s for s in snapshots["stages"] if s.get("status") == "COMPLETE"]
        stage = max(stages, key=lambda s: s.get("shuffleWriteBytes", 0) + s.get("shuffleReadBytes", 0))
        stage_id, attempt = stage["stageId"], stage["attemptId"]
        for endpoint, filename in ((f"stages/{stage_id}/{attempt}", "selected_stage.json"),
                                   (f"stages/{stage_id}/{attempt}/taskSummary", "selected_stage_task_summary.json")):
            (root / "ui_evidence" / filename).write_bytes(fetch(f"{api}/{endpoint}"))
        sqls = [s for s in snapshots["sql"] if s.get("status") == "COMPLETED"]
        # Choose a benchmark query for UI evidence.
        uncached_jobs = {j for t in timings if t["condition"] == "uncached" for j in json.loads(t["job_ids"])}
        candidates = [s for s in sqls if uncached_jobs.intersection(s.get("successJobIds", []))]
        selected_sql = max(candidates or sqls, key=lambda s: s.get("duration", 0))
        sql_id = selected_sql["id"]
        (root / "ui_evidence" / "selected_sql.json").write_bytes(fetch(f"{api}/sql/{sql_id}"))
        require(any(r.get("numCachedPartitions", 0) > 0 for r in snapshots["storage/rdd"]), "Cache was not materialized.")
        manifest["ui_metrics"] = {
            "selected_sql_id": sql_id, "selected_stage_id": stage_id, "selected_stage_attempt": attempt,
            "max_stage_shuffle_write_bytes": max(s.get("shuffleWriteBytes", 0) for s in stages),
            "total_memory_spill_bytes": sum(s.get("memoryBytesSpilled", 0) for s in stages),
            "total_disk_spill_bytes": sum(s.get("diskBytesSpilled", 0) for s in stages),
            "completed_stage_task_counts": sorted({s.get("numCompleteTasks", 0) for s in stages}),
        }
        for record in timings:
            jobs = [j for j in snapshots["jobs"] if j["jobId"] in json.loads(record["job_ids"])]
            stage_ids = {s for j in jobs for s in j.get("stageIds", [])}
            record["executed_stage_tasks"] = [{"stage_id": s["stageId"], "tasks": s["numCompleteTasks"]}
                                               for s in stages if s["stageId"] in stage_ids]
        manifest["trial_job_stage_links"] = timings

        # Spark UI: capture live pages with installed Chrome.
        chrome = args.chrome_binary or shutil.which("google-chrome") or shutil.which("chromium")
        mac_chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        if not chrome and mac_chrome.is_file():
            chrome = str(mac_chrome)
        views = [("01_sql_query_details.png", f"{ui}/SQL/execution/?id={sql_id}"),
                 ("02_stage_details.png", f"{ui}/stages/stage/?id={stage_id}&attempt={attempt}"),
                 ("03_storage_cache.png", f"{ui}/storage/")]
        for filename, url in views:
            evidence = {"filename": filename, "url": url, "status": "unavailable"}
            try:
                require(chrome, "No installed Chrome/Chromium binary found; use --keep-ui-open for manual capture.")
                html = fetch(url)
                (root / "ui_evidence" / filename.replace(".png", ".html")).write_bytes(html)
                path = root / "screenshots" / filename
                command = [chrome, "--headless", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                           f"--user-data-dir={root / 'chrome_profile'}", "--hide-scrollbars", "--window-size=1700,2400",
                           "--virtual-time-budget=5000", "--timeout=20000", f"--screenshot={path}", url]
                completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
                (root / "ui_evidence" / filename.replace(".png", "_chrome.log")).write_text(completed.stdout + completed.stderr)
                require(completed.returncode == 0 and path.is_file(), "Chrome screenshot command failed.")
                # Validate PNG integrity and decompression.
                png = path.read_bytes()
                require(png[:8] == b"\x89PNG\r\n\x1a\n", "Invalid PNG signature.")
                offset, compressed = 8, bytearray()
                width = height = 0
                while offset < len(png):
                    length = struct.unpack(">I", png[offset:offset+4])[0]
                    kind = png[offset+4:offset+8]
                    chunk = png[offset+8:offset+8+length]
                    crc = struct.unpack(">I", png[offset+8+length:offset+12+length])[0]
                    require(zlib.crc32(kind + chunk) & 0xffffffff == crc, "Corrupt PNG chunk.")
                    if kind == b"IHDR":
                        width, height = struct.unpack(">II", chunk[:8])
                    if kind == b"IDAT":
                        compressed.extend(chunk)
                    offset += length + 12
                require(width > 0 and height > 0 and len(zlib.decompress(compressed)) > 0, "Unreadable PNG.")
                evidence.update(status="captured", bytes=len(png), width=width, height=height, sha256=sha256(path))
                log(f"Captured {path.name} from {url}")
            except Exception as error:
                evidence["error"] = str(error)
                log(f"Automatic screenshot unavailable: {error}; live page: {url}")
            manifest["screenshots"].append(evidence)
            write_manifest(root, manifest)

        def summary(name, condition, partitions=4):
            return next(s for s in summaries if s["workload"] == name and s["condition"] == condition
                        and s["shuffle_partitions"] == partitions)
        comparisons = []
        for name in references:
            u, c, p = summary(name, "uncached"), summary(name, "cached"), summary(name, "uncached", 2)
            comparisons.append(f"{name}: uncached four-partition median {u['median_seconds']:.4f}s "
                               f"(range {u['min_seconds']:.4f}–{u['max_seconds']:.4f}s), "
                               f"cached median {c['median_seconds']:.4f}s, "
                               f"ratio {u['median_seconds']/c['median_seconds']:.2f}x; "
                               f"uncached two-partition median {p['median_seconds']:.4f}s.")
        uncached_total = sum(summary(n, "uncached")["total_seconds"] for n in references)
        cached_total = population + sum(summary(n, "cached")["total_seconds"] for n in references)
        manifest["cache"].update(uncached_repeated_total_seconds=uncached_total,
                                 cached_repeated_total_including_population_seconds=cached_total,
                                 paid_off_for_measured_repetitions=cached_total < uncached_total)
        plan = (root / "plans" / "filtered_recall_uncached_p4_after_formatted.txt").read_text()
        cached_plan = (root / "plans" / "filtered_recall_cached_p4_after_formatted.txt").read_text()
        scan_lines = [line.strip() for line in plan.splitlines() if any(t in line for t in ("PushedFilters:", "ReadSchema:", "PartitionFilters:"))]
        manifest["plan_observations"] = {"scan_details": scan_lines,
            "uncached_contains_exchange": "Exchange" in plan,
            "cached_contains_inmemory_scan": "InMemoryTableScan" in cached_plan,
            "adaptive_final_plan": "isFinalPlan=true" in plan}
        require("InMemoryTableScan" in cached_plan, "Cached query plan does not use cached data.")
        operator_discussion = (
            "The filtered plan uses partial/final HashAggregate operators and two Exchange boundaries "
            "for the grouped statistics plus distinct respondent count. The longitudinal plan uses "
            "Sort/SortAggregate for first/last structs followed by a summary HashAggregate. These "
            "grouping and shuffle steps remain work after column pruning; neither workload joins data. "
            "The longitudinal recall-not-null filter remains after mode expansion; proxy and complete-sleep "
            "filters reach its Parquet scan. AQEShuffleRead is marked coalesced in the final plans. "
        )
        paragraphs = [
            "Stage 2 reads the existing 198,704-row panel, not the 1.85 GB Stata file. "
            f"Its {manifest['input']['files']} Parquet files total {manifest['input']['bytes']:,} bytes. "
            "Projection and filters are applied before expansion/grouping. The actual filtered-recall scan reports: "
            + " | ".join(scan_lines) + ". These describe requested Parquet predicate pushdown and selected columns; "
            "they do not establish that row groups were skipped. No directory partition pruning is claimed. "
            "Stage 1's filters after wide-to-long expansion cannot be diagnosed from these Stage 2 plans.",
            "Saved before/after explain plans and UI REST metrics document physical execution. "
            f"The uncached filtered plan contains Exchange={manifest['plan_observations']['uncached_contains_exchange']}; "
            f"AQE final plan={manifest['plan_observations']['adaptive_final_plan']}. "
            f"Completed stages used task counts {manifest['ui_metrics']['completed_stage_task_counts']}; "
            f"the largest stage shuffle write was {manifest['ui_metrics']['max_stage_shuffle_write_bytes']:,} bytes, "
            f"with total memory/disk spill {manifest['ui_metrics']['total_memory_spill_bytes']:,}/"
            f"{manifest['ui_metrics']['total_disk_spill_bytes']:,} bytes. "
            + operator_discussion +
            "The longitudinal query requires respondent and summary grouping; its task and shuffle costs are in ui_evidence. "
            "The input is small, so these observations do not establish large-data scalability or a CPU/I/O bottleneck. "
            "Configured shuffle partitions may be coalesced by AQE; they are not guaranteed executed task counts.",
            " ".join(comparisons) + f" Shared cache population cost {population:.4f}s for {cached_rows:,} rows. "
            f"Across {args.repeats} repetitions of BOTH workloads, uncached actions totaled {uncached_total:.4f}s; "
            f"cache population plus cached actions totaled {cached_total:.4f}s, so caching "
            + ("paid off" if cached_total < uncached_total else "did not pay off")
            + " for this measured batch. Warm-ups and first runs are separately labeled in timings.csv and excluded from that comparison. "
            "Uncached means no Spark data cache, not cold OS storage; JVM warm-up, fixed run order, and local-machine noise limit causal interpretation. "
            "Cached plans and the live Storage screenshot demonstrate materialization. Stage 1's 160.65s includes conversion/startup/checksums and is not a query baseline.",
        ]
        (root / "performance_analysis.txt").write_text("\n\n".join(paragraphs) + "\n")
        if args.keep_ui_open:
            log(f"UI remains available at {ui}. Press Enter after inspecting/capturing evidence.")
            input()
        manifest["verification"] = {"all_trials_match_stage1": True,
            "stage1_and_protected_files_unchanged": all(p.is_file() and sha256(p) == before[str(p)] for p in protected)}
        require(manifest["verification"]["stage1_and_protected_files_unchanged"], "Protected files changed.")
        manifest["status"] = "complete" if all(s["status"] == "captured" for s in manifest["screenshots"]) else "complete_screenshots_unavailable"
        log("; ".join(comparisons))
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if spark is not None:
            try:
                if cached is not None:
                    cached.unpersist(blocking=True)
                spark.stop()
            except Exception as error:
                manifest["cleanup_error"] = str(error)
        manifest["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        manifest["event_log_files"] = [str(p.relative_to(root)) for p in (root / "event_logs").rglob("*") if p.is_file()]
        write_manifest(root, manifest)
        log(f"Status: {manifest['status']}; elapsed {manifest['elapsed_seconds']:.2f}s; evidence: {root}")


def run_stage3(args):
    """Stage 3 — Actions vs. Transformations."""
    import ast
    import contextlib
    import csv
    import io
    import math
    import pyarrow.parquet as pq
    from pyspark.sql import SparkSession, functions as F

    started = time.perf_counter()
    source = Path(args.stage1_dir).expanduser().resolve()
    raw_root = Path(args.output_dir).expanduser().absolute()
    root = raw_root.resolve()
    stage1 = json.loads((source / "run_manifest.json").read_text())
    require(stage1.get("status") == "complete", "Stage 1 must be complete.")
    protected_roots = {source, Path("output/stage1").resolve(), Path("output/stage2").resolve()}
    require(not raw_root.is_symlink() and root not in Path(__file__).resolve().parents,
            "Choose a dedicated Stage 3 output directory, not a symlink or project ancestor.")
    require(all(root != p and root not in p.parents and p not in root.parents for p in protected_roots),
            "Stage 3 output must not overlap Stage 1 or Stage 2.")
    owner = "ids706-pyspark-stage3-v1"
    if root.exists():
        require(args.overwrite, f"Output exists: {root}. Use --overwrite intentionally.")
        marker = root / "run_manifest.json"
        require(marker.is_file() and json.loads(marker.read_text()).get("owner") == owner,
                "Refusing overwrite: no Stage 3 ownership manifest.")
        allowed = {"plans", "event_logs", "spark_local", "warehouse", "execution_log.txt",
                   "operation_evidence.json", "summary_results.csv", "actions_vs_transformations.txt",
                   "run_manifest.json", "run_manifest.json.tmp"}
        require(all(p.name in allowed and not p.is_symlink() for p in root.iterdir()),
                "Refusing overwrite: unexpected files or symlinks.")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for name in ("plans", "event_logs"):
        (root / name).mkdir()
    protected = {p for folder in protected_roots for p in folder.rglob("*") if p.is_file()}
    protected.update(Path(n).resolve() for n in ("README.md", "requirements.txt", ".gitignore") if Path(n).is_file())
    before = {str(p): sha256(p) for p in sorted(protected)}
    script = Path(__file__).read_text()
    function_hashes = {n.name: hashlib.sha256(ast.get_source_segment(script, n).encode()).hexdigest()
                       for n in ast.parse(script).body if isinstance(n, ast.FunctionDef) and n.name in ("run_stage1", "run_stage2")}
    originals = [Path(stage1["source"]), Path("randhrs.parquet").resolve()]
    source_stats = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in originals if p.is_file()}
    manifest = {"owner": owner, "status": "incomplete", "arguments": vars(args),
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "protected_checksums_before": before, "stage_function_sha256": function_hashes,
                "original_dataset_stats_before": source_stats}
    evidence = []
    spark = None

    def log(message):
        text = f"[Stage 3] {message}"
        print(text, flush=True)
        with (root / "execution_log.txt").open("a") as stream:
            stream.write(text + "\n")

    def save_plans(frame, label):
        for mode in ("formatted", "extended"):
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                frame.explain(mode=mode)
            text = capture.getvalue()
            (root / "plans" / f"{label}_{mode}.txt").write_text(text)
            log(f"explain({mode}) [{label}]\n{text}")

    write_manifest(root, manifest)
    try:
        os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
        spark = (SparkSession.builder.master(args.master).appName("IDS706 Stage 3 Lazy Evaluation")
                 .config("spark.ui.enabled", "false")
                 .config("spark.ui.showConsoleProgress", "false")
                 .config("spark.eventLog.enabled", "true")
                 .config("spark.eventLog.dir", (root / "event_logs").as_uri())
                 .config("spark.sql.shuffle.partitions", args.shuffle_partitions)
                 .config("spark.sql.adaptive.enabled", "true")
                 .config("spark.sql.session.timeZone", "UTC")
                 .config("spark.sql.warehouse.dir", (root / "warehouse").as_uri())
                 .config("spark.local.dir", str(root / "spark_local"))
                 .config("spark.sql.artifact.isolation.enabled", "false").getOrCreate())
        spark.sparkContext.setLogLevel("ERROR")
        sc = spark.sparkContext
        tracker = sc.statusTracker()
        manifest["spark"] = {"version": spark.version, "master": sc.master, "application_id": sc.applicationId}

        def jobs(group):
            return sorted(tracker.getJobIdsForGroup(group))

        def record(operation, category, group, previous, elapsed=None, expect_jobs=False):
            # Poll for asynchronous action status updates.
            deadline = time.perf_counter() + 5
            current = jobs(group)
            while expect_jobs and not (set(current) - set(previous)) and time.perf_counter() < deadline:
                time.sleep(.05)
                current = jobs(group)
            new = sorted(set(current) - set(previous))
            details = []
            for job_id in new:
                info = tracker.getJobInfo(job_id)
                details.append({"job_id": job_id, "status": info.status if info else "not yet available",
                                "stage_ids": [int(s) for s in info.stageIds] if info else []})
            evidence.append({"operation": operation, "category": category, "job_group": group,
                             "job_ids_before": previous, "job_ids_after": current,
                             "new_job_ids": new, "elapsed_seconds": elapsed, "jobs": details})
            log(f"{operation}: {category}; before={previous}, after={current}, new={new}"
                + (f"; {elapsed:.4f}s" if elapsed is not None else ""))
            if category in ("transformation", "explain"):
                require(not new, f"Unexpected execution during {operation}: {new}")
            if expect_jobs:
                require(new, f"No Spark job evidence observed for {operation}.")

        # Track metadata loading separately from lazy transformations.
        group = "stage3:metadata_loading"
        sc.setJobGroup(group, "Stage 3: initial Parquet metadata/schema loading")
        previous = jobs(group)
        start = time.perf_counter()
        panel = spark.read.parquet(str(source / "respondent_wave"))
        schema = panel.schema
        record("Parquet load and schema resolution", "metadata_loading", group, previous, time.perf_counter()-start)
        manifest["input"] = {"path": str(source / "respondent_wave"), "columns": len(schema),
                             "stage1_rows": stage1["counts"]["age50_respondent_waves"]}

        group = "stage3:transformations"
        sc.setJobGroup(group, "Stage 3: construct transformations without actions")
        log("Constructing transformations; no count/show/collect is called in this section")
        previous = jobs(group)
        selected = panel.select("hhidpn", "wave", "age_band", "proxy", "frequent_sleep_trouble", "tr20_ftf_tel")
        record("select()", "transformation", group, previous)
        previous = jobs(group)
        filtered = selected.filter((F.col("wave") >= 12) & (F.col("proxy") == 0) &
                                   F.col("frequent_sleep_trouble").isNotNull() &
                                   F.col("tr20_ftf_tel").isNotNull())
        record("filter()", "transformation", group, previous)
        previous = jobs(group)
        transformed = filtered.withColumn("recall_fraction", F.col("tr20_ftf_tel") / F.lit(20.0))
        record("withColumn()", "transformation", group, previous)
        previous = jobs(group)
        keys = ["wave", "age_band", "frequent_sleep_trouble"]
        summary = transformed.groupBy(*keys).agg(F.count("*").alias("observations"),
                                               F.avg("recall_fraction").alias("mean_recall_fraction"))
        record("groupBy().agg()", "transformation", group, previous)
        previous = jobs(group)
        save_plans(summary, "before_actions")
        time.sleep(.2)  # Allow late status updates before the final lazy-evaluation check.
        record("explain()", "explain", group, previous)
        require(jobs(group) == [], "Transformation group submitted jobs.")

        # Actions: count and show trigger execution.
        group = "stage3:action_count"
        sc.setJobGroup(group, "Stage 3 action: count qualifying observations")
        previous = jobs(group)
        start = time.perf_counter()
        qualifying = filtered.count()
        record("count()", "action", group, previous, time.perf_counter()-start, expect_jobs=True)
        log(f"Qualifying respondent-wave observations: {qualifying:,}")

        group = "stage3:action_show"
        sc.setJobGroup(group, "Stage 3 action: show ten grouped summaries")
        previous = jobs(group)
        capture = io.StringIO()
        start = time.perf_counter()
        with contextlib.redirect_stdout(capture):
            summary.show(10, truncate=False)
        elapsed = time.perf_counter()-start
        log(capture.getvalue())
        record("show(10, truncate=False)", "action", group, previous, elapsed, expect_jobs=True)

        # Collect only bounded aggregates for validation and CSV output.
        group = "stage3:verification_collect"
        sc.setJobGroup(group, "Stage 3 supplemental action: collect at most 89 aggregate groups")
        previous = jobs(group)
        start = time.perf_counter()
        rows = [r.asDict() for r in summary.limit(89).collect()]
        record("limit(89).collect() for validation/output", "supplemental_action", group, previous,
               time.perf_counter()-start, expect_jobs=True)
        require(len(rows) <= 88, "Unexpected summary size; refusing unbounded output.")
        expected = [r for r in pq.read_table(source / "sleep_recall_summary").to_pylist()
                    if r["wave"] >= 12 and r["cognition_mode"] == "ftf_tel"]
        key = lambda r: tuple(r[k] for k in keys)
        expected_by_key = {key(r): r for r in expected}
        require({key(r) for r in rows} == set(expected_by_key), "Summary groups differ from Stage 1.")
        for row in rows:
            reference = expected_by_key[key(row)]
            require(row["observations"] == reference["observations"] and
                    math.isclose(row["mean_recall_fraction"], reference["mean_recall"] / 20,
                                 rel_tol=1e-10, abs_tol=1e-10), "Grouped statistics differ from Stage 1.")
        require(qualifying == sum(r["observations"] for r in expected) == sum(r["observations"] for r in rows),
                "Filtered counts differ from Stage 1 or summary denominators.")
        with (root / "summary_results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys + ["observations", "mean_recall_fraction"])
            writer.writeheader()
            writer.writerows(sorted(rows, key=key))
        # show/limit execute derived queries; the summary plan may remain non-final.
        save_plans(summary, "after_actions")
        explanation = (
            "select(), filter(), and withColumn() define projections, row restrictions, and a recall/20 expression. "
            "groupBy() creates a grouped object; agg() defines its aggregate DataFrame. These transformations "
            "build a logical plan without processing respondent rows. explain() displays logical/physical plans, "
            "not query results. The separately tracked metadata-loading group accounts for any initial Parquet "
            "schema-discovery jobs. The transformation group had no jobs, including after explain().\n\n"
            f"count() triggered execution and returned {qualifying:,} qualifying respondent-wave observations. "
            "show(10, truncate=False) triggered the grouped calculation and printed at most ten groups. "
            "Job IDs and action times are in operation_evidence.json; one action can launch multiple jobs. "
            "count() may prune unused expressions, so it is not evidence that every derived column was evaluated. "
            "The grouped mean explicitly consumes recall_fraction. No caching or timing speedup claim is used.\n\n"
            f"A separately labeled limit(89).collect() action returned {len(rows)} aggregate groups for validation "
            "and CSV output; it never collected respondent-level data. Counts and mean recall fractions match "
            "Stage 1's face-to-face/telephone summaries for waves 12–16, within floating-point tolerance. "
            "Web recall is not combined with this mode. Results are unweighted descriptions. Repeated actions "
            "can recompute work without persistence; their job counts need not be identical.\n"
        )
        (root / "actions_vs_transformations.txt").write_text(explanation)
        after = {str(p): sha256(p) for p in protected if p.is_file()}
        same_stats = all(p.is_file() and (p.stat().st_size, p.stat().st_mtime_ns) == source_stats[str(p)]
                         for p in originals if str(p) in source_stats)
        manifest["verification"] = {"transformation_job_ids": jobs("stage3:transformations"),
                                    "actions_submitted_jobs": True, "stage1_counts_match": True,
                                    "stage1_grouped_statistics_match": True,
                                    "protected_files_unchanged": before == after,
                                    "original_dataset_stats_unchanged": same_stats}
        require(before == after and same_stats, "Protected outputs or original dataset metadata changed.")
        manifest.update(status="complete", qualifying_observations=qualifying, summary_groups=len(rows))
        log(f"Verification passed: {qualifying:,} observations, {len(rows)} summary groups; protected files unchanged")
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if spark is not None:
            try:
                spark.sparkContext.setLocalProperty("spark.jobGroup.id", None)
                spark.sparkContext.setLocalProperty("spark.job.description", None)
                spark.stop()
            except Exception as error:
                manifest["cleanup_error"] = str(error)
        (root / "operation_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        manifest["elapsed_seconds"] = round(time.perf_counter()-started, 2)
        manifest["event_log_files"] = [str(p.relative_to(root)) for p in (root / "event_logs").rglob("*") if p.is_file()]
        write_manifest(root, manifest)
        log(f"Status: {manifest['status']}; elapsed {manifest['elapsed_seconds']:.2f}s; evidence: {root}")


def run_stage4(args):
    """Stage 4 — Machine Learning (Optional)."""
    import ast
    import csv
    import math
    import subprocess
    from pyspark import StorageLevel
    from pyspark.ml import Pipeline, PipelineModel
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
    from pyspark.ml.functions import vector_to_array
    from pyspark.ml.regression import LinearRegression
    from pyspark.sql import SparkSession, functions as F

    started = time.perf_counter()
    source = Path(args.stage1_dir).expanduser().resolve()
    raw_root = Path(args.output_dir).expanduser().absolute()
    root = raw_root.resolve()
    stage1 = json.loads((source / "run_manifest.json").read_text())
    require(stage1.get("status") == "complete", "Stage 1 must be complete.")
    protected_roots = {source, *(Path(f"output/stage{s}").resolve() for s in (1, 2, 3))}
    require(not raw_root.is_symlink() and root not in Path(__file__).resolve().parents,
            "Choose a dedicated Stage 4 directory, not a project ancestor or symlink.")
    require(all(root != p and root not in p.parents and p not in root.parents for p in protected_roots),
            "Stage 4 output must not overlap any previous stage.")
    owner = "ids706-pyspark-stage4-v1"
    if root.exists():
        require(args.overwrite, f"Output exists: {root}. Use --overwrite intentionally.")
        marker = root / "run_manifest.json"
        require(marker.is_file() and json.loads(marker.read_text()).get("owner") == owner,
                "Refusing overwrite: no Stage 4 ownership manifest.")
        allowed = {"model", "respondent_splits", "test_predictions", "coefficients.csv", "metrics.json",
                   "model_summary.txt", "execution_log.txt", "run_manifest.json", "run_manifest.json.tmp",
                   "spark_local", "warehouse"}
        require(all(p.name in allowed and not p.is_symlink() for p in root.iterdir()),
                "Refusing overwrite: unexpected files or symlinks.")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    protected = {p for folder in protected_roots for p in folder.rglob("*") if p.is_file()}
    protected.update(Path(n).resolve() for n in ("README.md", "requirements.txt", ".gitignore") if Path(n).is_file())
    before = {str(p): sha256(p) for p in sorted(protected)}
    script = Path(__file__).read_text()
    previous_functions = {n.name: hashlib.sha256(ast.get_source_segment(script, n).encode()).hexdigest()
                          for n in ast.parse(script).body if isinstance(n, ast.FunctionDef)
                          and n.name in ("run_stage1", "run_stage2", "run_stage3")}
    original_stats = {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
                      for p in (Path(stage1["source"]), Path("randhrs.parquet").resolve()) if p.is_file()}
    manifest = {"owner": owner, "status": "incomplete", "arguments": vars(args),
                "started_utc": datetime.now(timezone.utc).isoformat(), "stage_function_sha256": previous_functions,
                "protected_checksums_before": before, "original_dataset_stats_before": original_stats,
                "split": {"method": "pmod(xxhash64(hhidpn, lit(42)), 10000) < 8000", "seed": 42},
                "model_parameters": {"algorithm": "LinearRegression", "regParam": .1, "elasticNetParam": 0.,
                                     "standardization": True, "solver": "normal", "maxIter": 100},
                "interpretation": "Same-wave prediction for held-out respondents; unweighted, noncausal."}
    write_manifest(root, manifest)
    spark = None
    persisted = []

    def log(message):
        text = f"[Stage 4] {message}"
        print(text, flush=True)
        with (root / "execution_log.txt").open("a") as stream:
            stream.write(text + "\n")

    def finite(column):
        return column.isNotNull() & ~F.isnan(column) & (F.abs(column) != F.lit(float("inf")))

    try:
        os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
        spark = (SparkSession.builder.master(args.master).appName("IDS706 Stage 4 Recall Regression")
                 .config("spark.ui.enabled", "false").config("spark.ui.showConsoleProgress", "false")
                 .config("spark.sql.shuffle.partitions", args.shuffle_partitions)
                 .config("spark.sql.adaptive.enabled", "true").config("spark.sql.session.timeZone", "UTC")
                 .config("spark.sql.warehouse.dir", (root / "warehouse").as_uri())
                 .config("spark.local.dir", str(root / "spark_local"))
                 .config("spark.sql.artifact.isolation.enabled", "false").getOrCreate())
        spark.sparkContext.setLogLevel("ERROR")
        manifest["spark"] = {"version": spark.version, "master": spark.sparkContext.master,
                             "jvm_max_heap_bytes": spark._jvm.java.lang.Runtime.getRuntime().maxMemory()}
        log("Loading Stage 1 panel; no original conversion or previous stage is executed")
        label = "tr20_ftf_tel"
        categories = [*SLEEP, "wave"]
        columns = ["hhidpn", "wave", "age", "proxy", *SLEEP, label, "fimrc_ftf_tel", "fdlrc_ftf_tel"]
        panel = spark.read.parquet(str(source / "respondent_wave"))
        require(set(columns) <= set(panel.columns), "Missing required Stage 1 ML columns.")
        data = panel.select(*columns)
        counts = {"input_observations": data.count()}
        require(counts["input_observations"] == stage1["counts"]["age50_respondent_waves"], "Stage 1 dimensions differ.")
        # Record sequential exclusions; do not fill missing values.
        rules = [("nonproxy", F.col("proxy") == 0),
                 ("valid_observed_target", F.col(label).between(0, 20)),
                 ("not_imputed", (F.col("fimrc_ftf_tel") == 0) & (F.col("fdlrc_ftf_tel") == 0)),
                 ("valid_predictors", (F.col("age") >= 50) & F.col("wave").between(6, 16))]
        predictor_rule = rules[-1][1]
        for name in SLEEP:
            predictor_rule = predictor_rule & F.col(name).isin(1, 2, 3)
        rules[-1] = ("valid_predictors", predictor_rule)
        exclusions = []
        previous_count = counts["input_observations"]
        for name, rule in rules:
            data = data.filter(rule)
            retained = data.count()
            exclusions.append({"step": name, "excluded": previous_count-retained, "retained": retained})
            previous_count = retained
        manifest["exclusions"] = exclusions
        require(previous_count > 0, "No eligible observations after exclusions.")
        require(data.filter(F.col("hhidpn").isNull()).count() == 0, "Null respondent identifiers.")
        require(data.groupBy("hhidpn", "wave").count().filter("count > 1").count() == 0, "Duplicate respondent-wave keys.")
        # Prevent leakage: assign all waves of a respondent to one split.
        eligible = (data.withColumn("split_bucket", F.pmod(F.xxhash64("hhidpn", F.lit(42)), F.lit(10000)))
                    .withColumn("split", F.when(F.col("split_bucket") < 8000, "train").otherwise("test"))
                    .persist(StorageLevel.MEMORY_AND_DISK))
        persisted.append(eligible)
        train, test = eligible.filter("split = 'train'"), eligible.filter("split = 'test'")
        for name, frame in (("train", train), ("test", test)):
            result = frame.agg(F.count("*").alias("observations"), F.countDistinct("hhidpn").alias("respondents")).first().asDict()
            require(result["observations"] > 0 and result["respondents"] > 0, f"Empty {name} split.")
            counts[name] = result
        overlap = train.select("hhidpn").distinct().join(test.select("hhidpn").distinct(), "hhidpn", "inner").count()
        require(overlap == 0, "Respondent leakage across training and testing.")
        baseline_mean = train.agg(F.avg(label)).first()[0]
        require(math.isfinite(baseline_mean), "Nonfinite training baseline.")
        log(f"Exclusions: {exclusions}")
        log(f"Split counts: {counts}; respondent overlap: {overlap}")
        manifest["counts"] = counts

        # Reject unseen test categories to preserve reference-level interpretation.
        coverage = {}
        for name in categories:
            levels = sorted(r[0] for r in train.select(name).distinct().collect())
            unseen = test.select(name).distinct().join(train.select(name).distinct(), name, "left_anti").count()
            require(unseen == 0, f"Test has unseen {name} categories; categorical policy is fail-fast.")
            coverage[name] = levels
        manifest["training_category_levels"] = coverage
        manifest["unseen_category_policy"] = "Fail-fast: verify test levels exist in train; StringIndexer/OneHotEncoder handleInvalid=error."
        indexers = [StringIndexer(inputCol=n, outputCol=n+"_index", stringOrderType="alphabetAsc", handleInvalid="error")
                    for n in categories]
        encoder = OneHotEncoder(inputCols=[n+"_index" for n in categories],
                                outputCols=[n+"_encoded" for n in categories], dropLast=True, handleInvalid="error")
        assembler = VectorAssembler(inputCols=["age"]+[n+"_encoded" for n in categories],
                                    outputCol="features", handleInvalid="error")
        regression = LinearRegression(featuresCol="features", labelCol=label, predictionCol="prediction",
                                      **{k:v for k,v in manifest["model_parameters"].items() if k != "algorithm"})
        log("Fitting indexers, encoder, feature assembler, and ridge regression on training data only")
        model = Pipeline(stages=[*indexers, encoder, assembler, regression]).fit(train)
        predictions = model.transform(test).withColumn("baseline_prediction", F.lit(baseline_mean)).persist(StorageLevel.MEMORY_AND_DISK)
        train_predictions = model.transform(train).persist(StorageLevel.MEMORY_AND_DISK)
        persisted.extend([predictions, train_predictions])
        for name, frame in (("train", train_predictions), ("test", predictions)):
            invalid = frame.filter(~finite(F.col("prediction")) | F.exists(vector_to_array("features"), lambda x: ~finite(x))).count()
            require(invalid == 0, f"Nonfinite {name} feature vectors or predictions.")
            require(frame.count() == counts[name]["observations"], f"{name} transform dropped rows.")

        metrics = {"baseline_training_mean": baseline_mean, "model": {}, "baseline": {}}
        for name, prediction in (("model", "prediction"), ("baseline", "baseline_prediction")):
            for metric in ("rmse", "mae", "r2"):
                value = RegressionEvaluator(labelCol=label, predictionCol=prediction, metricName=metric).evaluate(predictions)
                require(math.isfinite(value), f"Nonfinite {name} {metric}.")
                metrics[name][metric] = value
            row = predictions.agg(F.avg(F.pow(F.col(prediction)-F.col(label), 2)).alias("mse"),
                                  F.avg(F.abs(F.col(prediction)-F.col(label))).alias("mae"),
                                  F.var_pop(label).alias("label_variance")).first()
            require(row.label_variance > 0, "Test target has no variance; R2 is undefined.")
            for metric, value in {"rmse": math.sqrt(row.mse), "mae": row.mae, "r2": 1-row.mse/row.label_variance}.items():
                require(math.isclose(value, metrics[name][metric], rel_tol=1e-8, abs_tol=1e-8), f"Metric reconciliation failed: {metric}")
        metrics["out_of_range_predictions"] = predictions.filter(~F.col("prediction").between(0,20)).count()
        metrics["rmse_improvement_over_baseline"] = metrics["baseline"]["rmse"]-metrics["model"]["rmse"]
        (root / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        log(f"Held-out test metrics: {metrics}")

        # Match coefficients to assembled feature metadata.
        fitted = model.stages[-1]
        attributes = predictions.schema["features"].metadata["ml_attr"]
        attrs = sorted([a for items in attributes["attrs"].values() for a in items], key=lambda a:a["idx"])
        coefficients = [float(c) for c in fitted.coefficients]
        require(len(attrs) == len(coefficients) == attributes["num_attrs"] and
                [a["idx"] for a in attrs] == list(range(len(coefficients))), "Coefficient feature alignment failed.")
        require(all(math.isfinite(c) for c in [*coefficients, fitted.intercept]), "Nonfinite model coefficients.")
        expected_names = ["age"]
        references = {}
        for name, indexer in zip(categories, model.stages[:len(categories)]):
            levels = list(indexer.labels)
            references[name] = levels[-1]
            expected_names.extend(f"{name}_encoded_{level}" for level in levels[:-1])
        require([a["name"] for a in attrs] == expected_names, "Encoded category coefficient names differ from fitted labels.")
        records = [{"index": a["idx"], "feature": a["name"], "coefficient": c} for a,c in zip(attrs,coefficients)]
        records.append({"index": -1, "feature": "intercept", "coefficient": float(fitted.intercept)})
        with (root / "coefficients.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["index", "feature", "coefficient"])
            writer.writeheader(); writer.writerows(records)
        manifest["reference_categories"] = references

        log("Saving model, respondent assignments, predictions, and checking reload")
        model.write().save(str(root / "model"))
        splits = eligible.select("hhidpn", "split_bucket", "split").distinct()
        splits.coalesce(args.shuffle_partitions).write.parquet(str(root / "respondent_splits"))
        output_columns = ["hhidpn", "wave", "age", *SLEEP, label, "prediction", "baseline_prediction"]
        predictions.select(*output_columns).coalesce(args.shuffle_partitions).write.parquet(str(root / "test_predictions"))
        reloaded = PipelineModel.load(str(root / "model"))
        original_predictions = predictions.select("hhidpn", "wave", F.col("prediction").alias("original_prediction"))
        reload_predictions = reloaded.transform(test).select("hhidpn", "wave", F.col("prediction").alias("reloaded_prediction"))
        delta = original_predictions.join(reload_predictions, ["hhidpn", "wave"], "full")
        max_delta = delta.agg(F.max(F.abs(F.col("original_prediction")-F.col("reloaded_prediction")))).first()[0]
        require(max_delta is not None and max_delta < 1e-10 and delta.filter(
            F.col("original_prediction").isNull() | F.col("reloaded_prediction").isNull()).count() == 0,
            "Saved-model reload predictions differ.")
        require(spark.read.parquet(str(root / "test_predictions")).count() == counts["test"]["observations"], "Prediction read-back count differs.")
        require(spark.read.parquet(str(root / "respondent_splits")).count() == sum(counts[n]["respondents"] for n in ("train","test")), "Split read-back count differs.")
        highest = sorted(records[:-1], key=lambda r:abs(r["coefficient"]), reverse=True)[:6]
        manifest["largest_absolute_coefficients"] = highest
        manifest["coefficient_count"] = len(coefficients)
        manifest["metrics"] = metrics
        summary = (
            f"MLlib ridge regression (regParam=0.1, L2 only), training-only preprocessing; {len(coefficients)} features. "
            f"Training: {counts['train']}; testing: {counts['test']}; overlapping respondents: {overlap}. "
            f"Deterministic respondent hash split with seed 42. Exclusions: {exclusions}.\n\n"
            f"Held-out model metrics: {metrics['model']}; training-mean baseline metrics: {metrics['baseline']}. "
            f"Baseline mean: {baseline_mean:.6f}. Out-of-range model predictions (not clipped): {metrics['out_of_range_predictions']}. "
            f"Largest absolute coefficients: {highest}. Age is per year; categorical coefficients are recall-point "
            f"contrasts to reference levels {references}. Categories follow alphabetic string ordering. "
            "Age and categorical coefficient magnitudes are in different units and are not a universal importance ranking.\n\n"
            "This is same-wave, unweighted prediction for unseen respondents, not future-wave forecasting or causal analysis. "
            "Complete cases with observed (not imputed) recall and no proxy are selected; this can introduce selection bias. "
            "Respondents with more eligible waves contribute more training/evaluation observations. Household members may occur "
            "in different splits. Education and sex were absent from the Stage 1 extract and are omitted. "
            "Sleep responses are categorical, including the opposite direction of feeling rested. Test categories were checked "
            "against training coverage; unseen categories fail explicitly. No test-set tuning or coefficient significance claims.\n"
        )
        (root / "model_summary.txt").write_text(summary)
        original_unchanged = all(Path(p).is_file() and (Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==v for p,v in original_stats.items())
        manifest["verification"] = {"respondent_overlap": overlap, "finite_features_and_predictions": True,
            "coefficient_alignment": True, "metric_reconciliation": True, "reload_max_absolute_difference": max_delta,
            "protected_files_unchanged": all(p.is_file() and sha256(p)==before[str(p)] for p in protected),
            "original_dataset_stats_unchanged": original_unchanged}
        require(manifest["verification"]["protected_files_unchanged"] and original_unchanged, "Protected files changed.")
        check = subprocess.run(["git","check-ignore","--quiet",str(root / "metrics.json")], capture_output=True)
        require(check.returncode == 0, "Stage 4 output is not Git-ignored.")
        manifest["output_git_ignored"] = True
        tracked = subprocess.run(["git","ls-files","output"], capture_output=True, text=True, check=True)
        manifest["already_tracked_output_files"] = tracked.stdout.splitlines()
        manifest["status"] = "complete"
        log(f"Verification passed; largest absolute coefficients: {highest}")
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        log(f"ERROR: {type(error).__name__}: {error}")
        raise
    finally:
        if spark is not None:
            try:
                for frame in reversed(persisted):
                    frame.unpersist()
                spark.stop()
            except Exception as error:
                manifest["cleanup_error"] = str(error)
        manifest["elapsed_seconds"] = round(time.perf_counter()-started, 2)
        write_manifest(root, manifest)
        log(f"Status: {manifest['status']}; elapsed {manifest['elapsed_seconds']:.2f}s; outputs: {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3, 4), required=True,
                        help="Stage to execute independently (1–4)")
    parser.add_argument("--input", default="randhrs1992_2022v2.dta")
    parser.add_argument("--output-dir", default=None, help="Default: output/stageN for the selected stage")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--master", default="local[2]", help="Local Spark master, e.g. local[2]")
    parser.add_argument("--shuffle-partitions", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="Replace only this script's owned output")
    parser.add_argument("--stage1-dir", default="output/stage1", help="Completed Stage 1 outputs for Stages 2–4")
    parser.add_argument("--repeats", type=int, default=5, help="Measured repetitions per Stage 2 condition")
    parser.add_argument("--keep-ui-open", action="store_true", help="Stage 2: wait for Enter before stopping Spark")
    parser.add_argument("--chrome-binary", help="Existing Chrome/Chromium executable for real UI screenshots")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = f"output/stage{args.stage}"
    if args.stage == 2 and args.repeats < 1:
        parser.error("Stage 2 repeats must be positive.")
    if args.chunk_size < 1 or args.shuffle_partitions < 1:
        parser.error("Chunk size and shuffle partitions must be positive.")
    if not (args.master == "local" or (args.master.startswith("local[") and args.master.endswith("]"))):
        parser.error("Use a local Spark master for this assignment.")
    try:
        {1: run_stage1, 2: run_stage2, 3: run_stage3, 4: run_stage4}[args.stage](args)
    except Exception as error:
        print(f"[Stage {args.stage}] ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
