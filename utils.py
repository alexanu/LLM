# Dictionary to hold different scenarios and their estimated output-to-input token ratios
token_ratio_estimates = {
    'email_response': 1.5,    # Outputs are generally 1.5 times longer than the inputs
    'content_summary': 0.8,   # Summaries are usually shorter than the original content
    'factual_answer': 0.5,    # Direct answers to factual questions tend to be concise
    'creative_story': 3.0,    # Creative stories may be much longer than the initial prompts
    'detailed_explanation': 20,  # Detailed explanations or complex answers can be much longer
    'limit': 0  # This will be treated specially to limit output tokens
}

# Maximum tokens for the 'limit' scenario
max_output_tokens = 100

def calculate_cost(text, model_name = DefaultModel, scenario='limit'):

    enc = tiktoken.encoding_for_model(model_name)
    input_tokens = enc.encode(text)   
    input_tokens_count = len(input_tokens) 

    # Determine the output token count based on the scenario
    if scenario == 'limit':
        output_token_count = max_output_tokens
    else:
        ratio = token_ratio_estimates.get(scenario, 1)  # Use 1 as a default ratio if the scenario is not found
        output_token_count = int(input_tokens_count * ratio)
    
    # Calculate the total token count (input + estimated output)
    total_token_count = input_tokens_count + output_token_count
    
    # Retrieve cost per million tokens for the model, assuming it can be a dictionary or a single value
    cost_per_million = ModelCosts[model_name]
    if isinstance(cost_per_million, dict):
        # Assuming the model has separate costs for input and output, typically not the case but for example
        input_cost_per_million = cost_per_million.get('input', 0)
        output_cost_per_million = cost_per_million.get('output', 0)
        total_cost = (input_tokens_count / 1_000_000) * input_cost_per_million + (output_token_count / 1_000_000) * output_cost_per_million
    else:
        total_cost = (total_token_count / 1_000_000) * cost_per_million
    
    return round(total_cost,3)