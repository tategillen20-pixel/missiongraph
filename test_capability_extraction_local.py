"""Manual OpenAI smoke test using a local, non-SAM sample description."""

from ai.capability_extractor import extract_capabilities

SAMPLE_DESCRIPTION = """
The contractor shall develop and demonstrate autonomous navigation
software for unmanned ground vehicles. The system shall use machine
learning to process sensor data, identify obstacles, and plan safe
navigation routes.
""".strip()


def main() -> None:
    """Analyze one local description with the configured OpenAI model."""
    extraction = extract_capabilities(
        title="Autonomous Navigation Software",
        description=SAMPLE_DESCRIPTION,
        notice_id="LOCAL-TEST-001",
    )
    print("Model:", extraction["model"])
    for capability in extraction["capabilities"]:
        print(
            capability["name"],
            capability["category"],
            capability["confidence"],
            repr(capability["evidence_quote"]),
        )


if __name__ == "__main__":
    main()
