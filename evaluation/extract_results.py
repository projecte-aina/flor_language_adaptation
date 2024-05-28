import os
import json
import pandas as pd


# Function to map dataset names to task categories and associated random results
def get_task_info(dataset_name):
    if 'belebele' in dataset_name:
        return 'Reading Comprehension (acc)', 25.00
    elif 'flores' in dataset_name:
        return 'Translation (bleu)', 'no-random'
    elif any(keyword in dataset_name for keyword in ['teca', 'xnli']):
        return 'NLI (acc)', 33.33
    elif any(keyword in dataset_name for keyword in ['xstorycloze', 'copa']):
        return 'Commonsense Reasoning (acc)', 50.00
    elif any(keyword in dataset_name for keyword in ['coqcat', 'xquad', 'catalanqa']):
        return 'QA (acc)', 'no-random'
    elif any(keyword in dataset_name for keyword in ['parafraseja', 'paws']):
        return 'Paraphrase Identification (acc)', 50.00
    return 'Other (acc)', 'no-random'

# Function to extract the language from the dataset name
def extract_language(dataset_name, selected_languages):
    for lang in selected_languages:
        if f"_{lang}" in dataset_name:
            return lang
        elif any(keyword in dataset_name for keyword in ['coqcat', 'teca', 'catalanqa', 'parafraseja']):
            return "ca"
    return 'unknown'
   
def read_json_files(base_dir, selected_languages):
    data = []
    for model_name in os.listdir(base_dir):
        model_dir = os.path.join(base_dir, model_name)
        if os.path.isdir(model_dir):
            for txt_file in os.listdir(model_dir):
                if txt_file.endswith('.txt'):
                    txt_path = os.path.join(model_dir, txt_file)
                    with open(txt_path, 'r') as file:
                        content = json.load(file)
                        for dataset_name, metrics in content.get('results', {}).items():
                            if 'acc' in metrics:
                                metric = round(metrics.get('acc', 0) * 100, 2)
                            elif 'f1' in metrics:
                                metric = round(metrics.get('f1', None), 2)
                            elif 'bleu' in metrics:
                                metric = round(metrics.get('bleu', None), 2)
                            # Filter datasets based on selected languages
                            language = extract_language(dataset_name, selected_languages)
                            if language in selected_languages:
                                task_category, random_result = get_task_info(dataset_name)
                                data.append({
                                    'dataset': dataset_name,
                                    'task_category': task_category,
                                    'model': model_name,
                                    'results': metric,
                                    'random': random_result,
                                    'language': language
                                })
    return data

def create_dataframe(data):
    df = pd.DataFrame(data)
    df_pivot = df.pivot_table(index=['language', 'task_category', 'dataset', 'random'], columns='model', values='results').reset_index()
    df_sorted = df_pivot.sort_values(by=['language', 'task_category', 'dataset'])
    return df_sorted

def main():
    base_dir = './results'  # Replace with the path to your base directory containing model folders
    selected_languages = ['ca', 'es', 'en'] #['ca', 'es', 'en']  # List of selected languages
    data = read_json_files(base_dir, selected_languages)
    df = create_dataframe(data)
    
    # Save the DataFrame to an Excel file
    output_file = 'results.xlsx'
    output_path = os.path.join(base_dir, output_file)
    df.to_excel(output_path, index=False)
    print(f'{output_file} has been saved to {output_path}')

if __name__ == "__main__":
    main()
