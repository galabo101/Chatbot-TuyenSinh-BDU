import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import json
from datetime import datetime
import re

class StaticPageCrawler:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        self.crawl_results = []
    
    def extract_slug(self, url: str) -> str:
        slug = url.rstrip('/').split('/')[-1].replace('.html', '')
        slug = re.sub(r'-\d+$', '', slug)
        return slug[:100]
    
    def fetch_page(self, url: str):
        response = requests.get(url, headers=self.headers, timeout=10)
        response.encoding = 'utf-8'
        return BeautifulSoup(response.text, 'html.parser')
    
    def extract_text(self, soup: BeautifulSoup) -> str:
        for tag in soup.find_all(['header', 'footer', 'script', 'style', 'nav']):
            tag.decompose()
        text = soup.get_text(separator='\n', strip=True)
        return '\n'.join(line.strip() for line in text.split('\n') if line.strip())
    
    def extract_tables(self, soup: BeautifulSoup) -> list:
        tables = []
        for idx, table in enumerate(soup.find_all('table'), 1):
            rows = []
            for tr in table.find_all('tr'):
                cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                if cells:
                    rows.append(cells)
            if rows:
                tables.append({'index': idx, 'rows': rows})
        return tables
    
    def save_tables(self, tables: list, slug: str, url: str, url_folder: str) -> list:
        if not tables:
            return []
        
        tables_dir = os.path.join(url_folder, 'tables')
        os.makedirs(tables_dir, exist_ok=True)
        saved = []
        
        for table in tables:
            filename = f"{slug}_table_{table['index']}.json"
            data = {
                'source_url': url,
                'rows': len(table['rows']),
                'cols': len(table['rows'][0]) if table['rows'] else 0,
                'data': table['rows']
            }
            with open(os.path.join(tables_dir, filename), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            saved.append(filename)
        
        return saved
    
    def download_images(self, soup: BeautifulSoup, base_url: str, url_folder: str) -> list:
        images_dir = os.path.join(url_folder, 'images')
        os.makedirs(images_dir, exist_ok=True)
        downloaded = []
        
        for img in soup.find_all('img'):
            img_url = img.get('src') or img.get('data-src')
            if not img_url:
                continue
            
            try:
                img_url = urljoin(base_url, img_url)
                response = requests.get(img_url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    filename = os.path.basename(img_url.split('?')[0]) or f"image_{len(downloaded)}.jpg"
                    filepath = os.path.join(images_dir, filename)
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    downloaded.append(filename)
            except:
                pass
        
        if not downloaded:
            try:
                os.rmdir(images_dir)
            except:
                pass
        
        return downloaded
    
    def crawl_url(self, url: str):
        slug = self.extract_slug(url)
        url_folder = os.path.join(self.output_dir, slug)
        os.makedirs(url_folder, exist_ok=True)
        
        print(f"Crawling: {url}")
        
        try:
            soup = self.fetch_page(url)
            
            text = self.extract_text(soup)
            with open(os.path.join(url_folder, f"{slug}.txt"), 'w', encoding='utf-8') as f:
                f.write(text)
            
            tables = self.extract_tables(soup)
            saved_tables = self.save_tables(tables, slug, url, url_folder)
            
            images = self.download_images(soup, url, url_folder)
            
            self.crawl_results.append({
                'url': url,
                'folder': slug,
                'status': 'success',
                'text': len(text),
                'tables': len(saved_tables),
                'images': len(images)
            })
            print(f"  ✓ Text: {len(text)} | Tables: {len(saved_tables)} | Images: {len(images)}")
            
        except Exception as e:
            self.crawl_results.append({'url': url, 'status': 'error', 'error': str(e)})
            print(f"  ✗ Error: {str(e)}")
    
    def crawl_multiple(self, urls: list):
        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}]", end=' ')
            self.crawl_url(url)
        
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(self.crawl_results, f, ensure_ascii=False, indent=2)
        
        success = len([r for r in self.crawl_results if r['status'] == 'success'])
        print(f"\n{'='*60}\nDone: {success}/{len(urls)} success")


if __name__ == "__main__":
    OUTPUT_DIR = r"C:\Users\nguye\OneDrive\Desktop\New folder (2)\bo_sung"
    URL_FILE = r"C:\Users\nguye\OneDrive\Desktop\New folder (2)\url.txt"
    
    with open(URL_FILE, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"{'='*60}\nCrawling {len(urls)} URLs\n{'='*60}")
    crawler = StaticPageCrawler(OUTPUT_DIR)
    crawler.crawl_multiple(urls)