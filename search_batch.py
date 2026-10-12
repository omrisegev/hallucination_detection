import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import time

papers = [
    "Detecting Contextual Hallucinations in Large Language Models with Frequency-Aware Attention",
    "Mind the Gap: Catching Hallucinations via Evidence Drop on the Reasoning Manifold",
    "Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Model",
    "How do LLMs Compute Verbal Confidence?",
    "Inference-Time Conformal Reasoning with Valid Factuality Control for Large Language Models",
    "GAUSS: Graph-Assisted Uncertainty Quantification using Structure and Semantics for Long-Form Generation in LLMs",
    "HaloProbe: Bayesian Detection and Mitigation of Object Hallucinations in Vision-Language Models",
    "REVIS: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models",
    "Adaptive Residual-Update Steering for Low-Overhead Hallucination Mitigation in Large Vision-Language Models",
    "Agentic Confidence Calibration"
]

results = []

for paper in papers:
    # Build query
    query = f'ti:"{paper}"'
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": query,
        "max_results": 1
    })
    
    print(f"Searching for: {paper}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()
        
        root = ET.fromstring(xml_data)
        namespaces = {'atom': 'http://www.w3.org/2005/Atom'}
        entry = root.find('atom:entry', namespaces)
        
        if entry is not None:
            id_url = entry.find('atom:id', namespaces).text
            arxiv_id = id_url.split('/abs/')[-1].split('v')[0]
            title = entry.find('atom:title', namespaces).text.strip()
            summary = entry.find('atom:summary', namespaces).text.strip()
            pdf_url = entry.find('atom:link[@title="pdf"]', namespaces)
            pdf_link = pdf_url.attrib['href'] if pdf_url is not None else ""
            
            results.append({
                "query_title": paper,
                "arxiv_id": arxiv_id,
                "title": title,
                "pdf_url": pdf_link,
                "found": True
            })
            print(f"  Found arXiv ID: {arxiv_id}")
        else:
            # Let's try searching without quotes
            url_fallback = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({
                "search_query": f'all:"{paper}"',
                "max_results": 1
            })
            with urllib.request.urlopen(urllib.request.Request(url_fallback, headers={'User-Agent': 'Mozilla/5.0'})) as response:
                xml_data = response.read()
            root = ET.fromstring(xml_data)
            entry = root.find('atom:entry', namespaces)
            if entry is not None:
                id_url = entry.find('atom:id', namespaces).text
                arxiv_id = id_url.split('/abs/')[-1].split('v')[0]
                title = entry.find('atom:title', namespaces).text.strip()
                results.append({
                    "query_title": paper,
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "found": True
                })
                print(f"  Found arXiv ID (fallback): {arxiv_id}")
            else:
                results.append({
                    "query_title": paper,
                    "found": False
                })
                print(f"  Not found on arXiv.")
    except Exception as e:
        print(f"  Error: {e}")
        results.append({
            "query_title": paper,
            "found": False,
            "error": str(e)
        })
    
    # Rate limit: 1 request every 3 seconds
    time.sleep(3)

with open("arxiv_search_batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Finished batch search.")
