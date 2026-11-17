from pathlib import Path
from face_engine import embeddings_from_image

# CHANGE THIS to the name of one of your uploaded event photos
IMAGE = "uploads/events/1/0a210b82f8d24d66838dcfe414455eb0_WhatsApp_Image_2026-11-15_at_5.43.32_PM_1.jpeg"
print("Testing:", IMAGE)

embeddings = embeddings_from_image(IMAGE)

print()
print("================================")
print("TEST COMPLETE")
print("Faces with embeddings:", len(embeddings))

for i, embedding in enumerate(embeddings, start=1):
    print(
        f"Face {i}: "
        f"dimension={len(embedding)}, "
        f"norm={sum(x*x for x in embedding) ** 0.5:.4f}"
    )

print("================================")