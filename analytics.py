import pandas as pd


def compute_analytics(csv_path):

    df = pd.read_csv(csv_path)

    total_students = df["student_id"].nunique()

    activity_counts = (
        df.groupby("class_name")["student_id"]
        .count()
        .sort_values(ascending=False)
    )

    timeline = (
        df.groupby(["frame", "class_name"])
        .size()
        .unstack(fill_value=0)
    )

    return {
        "total_students": total_students,
        "activity_counts": activity_counts,
        "timeline": timeline,
        "raw_df": df,
    }
