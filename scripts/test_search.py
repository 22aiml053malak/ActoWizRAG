import asyncio
from app.repositories.vector_repository import VectorRepository
from app.services.embedding_service import EmbeddingService

async def main():
    repo = VectorRepository()
    embed = EmbeddingService()
    print("Embedding query...")
    q = embed.embed_query("give me assigment overview")
    print("Query len:", len(q))
    print("Searching...")
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(None, lambda: repo.search(q, similarity_top_k=5))
    print(f"Got {len(res)} results:")
    for r in res:
        print(r.node.text[:50])

if __name__ == "__main__":
    asyncio.run(main())
