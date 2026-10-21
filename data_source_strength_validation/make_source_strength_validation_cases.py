from __future__ import annotations

import csv
from pathlib import Path


# These centers lie halfway between the 0.125 m source-location grid used by
# data_u0p2_dense_969. They therefore test both source-strength scaling and
# spatial interpolation without adding a second CFD campaign.
POSITIONS = [
    (0.8125, 0.8125),
    (1.1875, 1.4375),
    (1.6875, 2.1875),
    (2.0625, 0.6875),
    (2.5625, 1.5625),
    (2.9375, 2.3125),
    (3.4375, 0.9375),
    (3.9375, 1.6875),
    (4.3125, 2.1875),
    (4.8125, 0.6875),
    (5.1875, 1.4375),
    (5.6875, 2.3125),
    (6.1875, 0.9375),
    (6.6875, 1.6875),
    (7.1875, 2.1875),
]

# Run q=1 first at each location; it is the reference field for the linearity
# checks C(q) ~= q*C(1). The other two levels bracket the training condition.
SOURCE_STRENGTHS = (1.0, 0.5, 2.0)
INLET_VELOCITY = 0.2
FIRST_CASE_ID = 950001


def main() -> None:
    output_path = Path(__file__).with_name("cases.csv")
    rows = []
    case_id = FIRST_CASE_ID

    for src_x, src_y in POSITIONS:
        for source_strength in SOURCE_STRENGTHS:
            rows.append(
                {
                    "case_id": case_id,
                    "inlet_u": f"{INLET_VELOCITY:.4f}",
                    "src_x": f"{src_x:.4f}",
                    "src_y": f"{src_y:.4f}",
                    "source_strength": f"{source_strength:.4f}",
                }
            )
            case_id += 1

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("case_id", "inlet_u", "src_x", "src_y", "source_strength"),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} cases to {output_path}")


if __name__ == "__main__":
    main()
