import pandas as pd


def compute_analytics(csv_path):

    df = pd.read_csv(csv_path)

    # =========================================================
    # ⏱ DURATION CALCULATION (FIRST — used for filtering)
    # =========================================================

    df = df.sort_values(["student_id", "timestamp"])

    df["time_diff"] = df.groupby("student_id")["timestamp"].diff()
    df["time_diff"] = df["time_diff"].clip(lower=0, upper=1)

    student_activity_duration = (
        df.groupby(["student_id", "class_name"])["time_diff"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "class_name": "activity",
                "time_diff": "duration_seconds",
            }
        )
    )

    MIN_DURATION = 1.0

    valid_students = student_activity_duration[
        student_activity_duration["duration_seconds"] >= MIN_DURATION
    ]["student_id"].unique()

    # =========================================================
    # ✅ FILTER MAIN DF USING VALID STUDENTS ONLY
    # =========================================================

    df = df[df["student_id"].isin(valid_students)]

    # =========================================================
    # 👥 REAL TOTAL STUDENTS (PEAK CONCURRENT COUNT)
    # =========================================================

    total_students = (
        df.groupby("frame")["student_id"]
        .nunique()
        .max()
    )

    # =========================================================
    # 📊 ACTIVITY COUNTS
    # =========================================================

    activity_counts = (
        df.groupby("class_name")["student_id"]
        .count()
        .sort_values(ascending=False)
    )

    activity_distribution = (
        df.groupby("class_name")
        .agg(
            frames=("frame", "count"),
            students=("student_id", lambda x: ", ".join(map(str, sorted(x.unique())))),
        )
        .reset_index()
    )

    timeline = (
        df.groupby(["frame", "class_name"])
        .size()
        .unstack(fill_value=0)
    )

    # round only for display
    student_activity_duration = student_activity_duration[
        student_activity_duration["student_id"].isin(valid_students)
    ]

    student_activity_duration["duration_seconds"] = (
        student_activity_duration["duration_seconds"].round(2)
    )

    return {
        "total_students": int(total_students),
        "activity_counts": activity_counts,
        "activity_distribution": activity_distribution,
        "timeline": timeline,
        "student_activity_duration": student_activity_duration,
        "raw_df": df,
    }
