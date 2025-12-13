"""
Web Search Corrector - Google Custom Search API Integration
Fallback mechanism when internal documents are insufficient
"""

from typing import List, Dict
import os


class WebSearchCorrector:    
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.cse_id = os.getenv("GOOGLE_CSE_ID")
        self.enabled = bool(self.api_key and self.cse_id)
    def search(self, query: str, max_results: int = 3) -> List[Dict]:

        print(f"[WebSearch] Searching: {query}")
        
        if not self.enabled:
            return []
        
        try:
            from googleapiclient.discovery import build
            
            # Build Custom Search service
            service = build("customsearch", "v1", developerKey=self.api_key)
            
            # Execute search
            result = service.cse().list(
                q=query,
                cx=self.cse_id,
                num=max_results,
                lr="lang_vi",
                gl="vn"
            ).execute()
            
            items = result.get("items", [])
            
            if not items:
                print("[WebSearch] No results found")
                return []
            
            # Format as chunks
            chunks = []
            for i, item in enumerate(items):
                snippet = item.get("snippet", "")
                
                # Try to get more content from pagemap
                pagemap = item.get("pagemap", {})
                metatags = pagemap.get("metatags", [{}])[0]
                description = metatags.get("og:description", snippet)
                
                chunk = {
                    "chunk_id": f"google_{i}_{item.get('cacheId', i)}",
                    "content": snippet[:500],
                    "full_content": f"{item.get('title', '')}\n\n{description}",
                    "url": item.get("link", ""),
                    "type": "web_search",
                    "title": item.get("title", ""),
                    "score": 0.70,
                    "source": "google_cse"
                }
                chunks.append(chunk)
            
            print(f"[WebSearch] ✅ Found {len(chunks)} results")
            return chunks
            
        except Exception as e:
            print(f"[WebSearch] ❌ Error: {e}")
            return []    
