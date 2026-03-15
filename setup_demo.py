"""
setup_demo.py — Run this ONCE before the demo to preload faces + interactions.

Place photos named after the person in the same folder:
  sarah.jpg, john.jpg, etc.

Then run:
  python setup_demo.py
"""

import os
from dotenv import load_dotenv
from face_handler import FaceMatcher
from interaction_store import InteractionStore

load_dotenv()

COLLECTION_ID = os.getenv("REKOGNITION_COLLECTION", "memoire-faces")
BUCKET_NAME   = os.getenv("S3_BUCKET", "memoire-faces-demo")
TABLE_NAME    = os.getenv("DYNAMODB_TABLE", "interactions")

# ── EDIT THIS: your demo people ──────────────────────────
DEMO_DATA = [
    {
        "person_id": "Yajat Mathur",
        "photo": "Yajat.jpeg",
        "interactions": [
            "Visited today and brought flowers. We talked about her upcoming wedding.",
            "Called to check in. She mentioned she is redecorating her new home.",
            "Had lunch together at the garden cafe. She seemed happy and laughed a lot.",
        ],
    },
    {
        "person_id": "Viren James",
        "photo": "viren.jpeg",
        "interactions": [
            "Dropped by after work. We watched the cricket match together.",
            "Brought homemade food. We talked about old family trips to Ooty.",
            "Called in the morning to wish happy birthday.",
        ],
    },
]
# ─────────────────────────────────────────────────────────


def main():
    face_matcher = FaceMatcher(COLLECTION_ID, BUCKET_NAME)
    store = InteractionStore(TABLE_NAME)

    for person in DEMO_DATA:
        pid   = person["person_id"]
        photo = person["photo"]

        # Index face
        if os.path.exists(photo):
            with open(photo, "rb") as f:
                image_bytes = f.read()
            try:
                face_matcher.index_face(image_bytes, pid)
                print(f"✅ Indexed face for: {pid}")
            except Exception as e:
                print(f"⚠️  Could not index face for {pid}: {e}")
        else:
            print(f"⚠️  Photo not found: {photo} — skipping face index for {pid}")

        # Add interactions
        for desc in person["interactions"]:
            store.add_interaction(pid, desc)
        print(f"✅ Loaded {len(person['interactions'])} interactions for: {pid}")

    print("\n🎉 Demo setup complete! Run: python main.py")


if __name__ == "__main__":
    main()
