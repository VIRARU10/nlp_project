def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list:
    """
    Splits text into sliding windows of words.
    
    Args:
        text (str): The string to chunk.
        chunk_size (int): The number of words per chunk.
        overlap (int): The number of words to overlap between chunks.
        
    Returns:
        list of str: The chunked strings.
    """
    if not text:
        return []
        
    words = text.split()
    if not words:
        return []
        
    if len(words) <= chunk_size:
        return [text]
        
    chunks = []
    step = max(1, chunk_size - overlap)
    
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        
        # if the remaining words are perfectly covered, break
        if i + chunk_size >= len(words):
            break
            
    return chunks
