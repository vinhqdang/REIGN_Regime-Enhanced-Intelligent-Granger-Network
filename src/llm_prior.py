import os
import json
import asyncio
import numpy as np
from anthropic import AsyncAnthropic

async def query_llm(client: AsyncAnthropic, prompt: str, model: str = "claude-3-5-sonnet-20241022", max_tokens: int = 10) -> float:
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()
        return float(text)
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return 0.5 # fallback

async def batch_query_pairs(client: AsyncAnthropic, pairs: list, domain_desc: str, repetitions: int = 3) -> dict:
    results = {}
    
    async def process_pair(i, j, vi, vidx, vj, vjdx):
        prompt = f"""Given the following business/financial variables in the context of {domain_desc}:
Variable i: {vi} — {vidx}
Variable j: {vj} — {vjdx}

On a scale from 0 to 1, how likely is it that
changes in {vi} DIRECTLY CAUSE changes in {vj}
in a financial/business context?

Respond with a single float between 0 and 1 only.
Do not explain."""
        
        scores = []
        for _ in range(repetitions):
            score = await query_llm(client, prompt)
            scores.append(score)
            
        mean_score = np.mean(scores)
        var_score = np.var(scores)
        return (i, j), scores, mean_score, var_score

    tasks = []
    for (i, j, vi, vidx, vj, vjdx) in pairs:
        tasks.append(process_pair(i, j, vi, vidx, vj, vjdx))
        
    batch_size = 50
    for idx in range(0, len(tasks), batch_size):
        batch = tasks[idx:idx+batch_size]
        batch_results = await asyncio.gather(*batch)
        for (i, j), scores, mean_score, var_score in batch_results:
            results[f"{i},{j}"] = {
                "scores": scores,
                "mean": float(mean_score),
                "var": float(var_score)
            }
            
    return results

def get_topological_order(variable_names: list, domain_desc: str) -> list:
    # Simplified placeholder for topological ordering extraction
    return variable_names

def apply_causal_order_correction(A_LLM: np.ndarray, variable_names: list, domain_desc: str) -> np.ndarray:
    order = get_topological_order(variable_names, domain_desc)
    order_dict = {name: idx for idx, name in enumerate(order)}
    
    A_corrected = A_LLM.copy()
    N = len(variable_names)
    for i in range(N):
        for j in range(N):
            if order_dict[variable_names[j]] < order_dict[variable_names[i]]:
                A_corrected[i, j] = 0.0
                
    return A_corrected

def generate_llm_prior(variable_names: list, variable_descriptions: list, domain_desc: str, dataset_name: str = "default", use_mock: bool = False) -> tuple:
    """
    Stage 2b: LLM Prior Generation
    Returns A_LLM matrix and confidence matrix.
    """
    N = len(variable_names)
    A_LLM = np.zeros((N, N))
    confidence_matrix = np.ones((N, N))
    
    is_unnamed = all(name.startswith("var_") for name in variable_names)
    if is_unnamed:
        print("Warning: Variable names are uninformative. Using uniform prior.")
        A_LLM = 0.5 * np.ones((N, N)) - 0.5 * np.eye(N)
        return A_LLM, confidence_matrix
        
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{dataset_name}_llm_prior.json")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            results = json.load(f)
    else:
        if use_mock:
            results = {}
            for i in range(N):
                for j in range(N):
                    if i != j:
                        results[f"{i},{j}"] = {"scores": [0.5, 0.5, 0.5], "mean": 0.5, "var": 0.0}
        else:
            client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
            pairs = []
            for i in range(N):
                for j in range(N):
                    if i != j:
                        pairs.append((i, j, variable_names[i], variable_descriptions[i], 
                                      variable_names[j], variable_descriptions[j]))
                        
            try:
                loop = asyncio.get_running_loop()
                print("Event loop running, using mock LLM response for now.")
                results = {}
                for (i, j, _, _, _, _) in pairs:
                    results[f"{i},{j}"] = {"scores": [0.5, 0.5, 0.5], "mean": 0.5, "var": 0.0}
            except RuntimeError:
                results = asyncio.run(batch_query_pairs(client, pairs, domain_desc))
                
            with open(cache_file, 'w') as f:
                json.dump(results, f)
                
    for key, val in results.items():
        i, j = map(int, key.split(','))
        A_LLM[i, j] = val["mean"]
        if val["var"] > 0.1:
            confidence_matrix[i, j] = 0.5 # weight halved
            
    max_val = np.max(A_LLM)
    if max_val > 0:
        A_LLM = A_LLM / max_val
        
    A_LLM = apply_causal_order_correction(A_LLM, variable_names, domain_desc)
    
    return A_LLM, confidence_matrix
