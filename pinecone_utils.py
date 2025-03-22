import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def get_pinecone_index_info(user):
    """Fetch metadata about the user's Pinecone index, including namespaces and filenames."""
    index_name = f"{user}-index"

    try:
        index = pc.Index(index_name)
        index_stats = index.describe_index_stats()
        namespaces = index_stats.get("namespaces", {})
        
        detailed_info = {}
        for namespace in namespaces.keys():
            response = index.query(
                vector=[0] * 768,  # Dummy vector
                top_k=1000,  # Increased limit to get more metadata
                include_metadata=True,
                namespace=namespace
            )
            filenames = {}
            for match in response.get('matches', []):
                if 'metadata' in match and 'filename' in match['metadata']:
                    filename = match['metadata']['filename']
                    if filename not in filenames:
                        filenames[filename] = []
                    filenames[filename].append(match['metadata']['vector_id'])
            detailed_info[namespace] = {
                'vector_count': namespaces[namespace]['vector_count'],
                'files': {filename: {'vector_ids': ids} for filename, ids in filenames.items()}
            }

        return {
            "user": user,
            "index_name": index_name,
            "namespaces": detailed_info,
            "total_vectors": index_stats.get("total_vector_count", 0),
        }
    except Exception as e:
        print(f"Error fetching index info: {e}")
        return {"error": str(e)}
    
def delete_file_vectors(user, namespace, filename=None):
    """Delete vectors associated with a specific file or all vectors in a namespace."""
    index_name = f"{user}-index"
    try:
        index = pc.Index(index_name)
        
        if filename:
            # Delete vectors for a specific file
            response = index.query(
                vector=[0] * 768,  # Dummy vector
                top_k=10000,  # Large enough to get all vectors
                include_metadata=True,
                namespace=namespace
            )
            
            ids_to_delete = []
            for match in response.get('matches', []):
                if 'metadata' in match and match['metadata'].get('filename') == filename:
                    ids_to_delete.append(match['id'])
            
            if ids_to_delete:
                index.delete(ids=ids_to_delete, namespace=namespace)
                return True, f"Deleted {len(ids_to_delete)} vectors for file '{filename}' in namespace '{namespace}'"
            else:
                return False, f"No vectors found for file '{filename}' in namespace '{namespace}'"
        else:
            # Delete all vectors in the namespace
            index.delete(delete_all=True, namespace=namespace)
            return True, f"Deleted all vectors in namespace '{namespace}' for user '{user}'"
            
    except Exception as e:
        print(f"Error deleting vectors: {e}")
        return False, str(e)