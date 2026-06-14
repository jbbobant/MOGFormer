import os
import mygene

def convert_entrez_to_symbols(input_file, output_file):
    # Read the Entrez gene IDs
    print(f"Reading Entrez IDs from {input_file}...")
    with open(input_file, 'r') as f:
        entrez_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(entrez_ids)} gene IDs. Querying MyGene.info...")
    
    # Initialize mygene client
    mg = mygene.MyGeneInfo()
    
    # Query mygene using querymany
    # scopes='entrezgene' indicates the input values are Entrez Gene IDs.
    # fields='symbol' requests the standard gene symbol in return.
    # species='human' targets human genes (e.g., Entrez ID 5473 is human PPIA).
    results = mg.querymany(
        entrez_ids, 
        scopes='entrezgene', 
        fields='symbol', 
        species='human',
        verbose=False
    )
    
    # mygene's querymany may return multiple hits or omit not-found items.
    # To preserve the exact ordering and length, we construct a mapping dictionary.
    symbol_map = {}
    for r in results:
        query_id = r.get('query')
        symbol = r.get('symbol')
        # Map query_id to its symbol. If duplicate matches occur, we keep the first valid one.
        if query_id and symbol and query_id not in symbol_map:
            symbol_map[str(query_id)] = symbol
            
    # Map the original list to symbols in the exact original order.
    # If a gene ID was not found, we fall back to the original Entrez ID.
    converted_symbols = []
    missing_count = 0
    for eid in entrez_ids:
        symbol = symbol_map.get(eid)
        if symbol:
            converted_symbols.append(symbol)
        else:
            converted_symbols.append(eid)
            missing_count += 1
            
    print(f"Mapping complete. {len(converted_symbols)} items mapped.")
    print(f"Missing symbols for {missing_count} genes (kept original Entrez ID in their place).")
    
    # Write to output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for symbol in converted_symbols:
            f.write(f"{symbol}\n")
            
    print(f"Successfully saved converted list to: {output_file}")

if __name__ == "__main__":
    # Define absolute or relative paths
    input_path = os.path.join("data", "raw", "FROGS-ARCHS4", "FROGS-ARCHS4_genelist.txt")
    output_path = os.path.join("data", "raw", "FROGS-ARCHS4", "FROGS-ARCHS4_symbollist.txt")
    convert_entrez_to_symbols(input_path, output_path)
