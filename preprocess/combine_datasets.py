import os
import json
import h5py
import argparse
from tqdm import tqdm

def combine_h5_files(mr_hisum_path, mr_ali_path, output_path):
    """
    Combina los archivos H5 de mr_hisum y mr_ali, manteniendo la estructura de mr_hisum
    """
    print("Combinando archivos H5...")
    
    # Abrir archivos H5
    with h5py.File(mr_hisum_path, 'r') as hisum_file, \
         h5py.File(mr_ali_path, 'r') as ali_file, \
         h5py.File(output_path, 'w') as output_file:
        
        # Copiar todos los grupos de mr_hisum
        print("Copiando datos de mr_hisum...")
        for key in tqdm(hisum_file.keys()):
            hisum_file.copy(key, output_file)
        
        # Copiar todos los grupos de mr_ali
        print("Copiando datos de mr_ali...")
        for key in tqdm(ali_file.keys()):
            # Verificar si la clave ya existe
            if key in output_file:
                print(f"Advertencia: La clave {key} ya existe en mr_hisum. Se omitirá.")
                continue
            ali_file.copy(key, output_file)

def combine_json_files(mr_hisum_json_path, mr_ali_json_path, output_json_path):
    """
    Combina los archivos JSON de splits, manteniendo la estructura de mr_hisum
    """
    print("Combinando archivos JSON...")
    
    # Cargar archivos JSON
    with open(mr_hisum_json_path, 'r') as f:
        hisum_data = json.load(f)
    
    with open(mr_ali_json_path, 'r') as f:
        ali_data = json.load(f)
    
    # Crear nuevo diccionario con la estructura de mr_hisum
    combined_data = {
        "train_keys": hisum_data.get("train_keys", []),
        "test_keys": hisum_data.get("test_keys", []),
        "val_keys": hisum_data.get("val_keys", [])
    }
    
    # Agregar claves de mr_ali
    combined_data["train_keys"].extend(ali_data.get("train", []))
    combined_data["test_keys"].extend(ali_data.get("test", []))
    combined_data["val_keys"].extend(ali_data.get("val", []))
    
    # Guardar archivo combinado
    with open(output_json_path, 'w') as f:
        json.dump(combined_data, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Combinar datasets mr_hisum y mr_ali')
    parser.add_argument('--mr_hisum_h5', default='../dataset/mr_hisum.h5', help='Ruta al archivo H5 de mr_hisum')
    parser.add_argument('--mr_ali_h5', default='../dataset/mr_ali.h5', help='Ruta al archivo H5 de mr_ali')
    parser.add_argument('--mr_hisum_json', default='../dataset/mr_hisum_split.json', help='Ruta al archivo JSON de splits de mr_hisum')
    parser.add_argument('--mr_ali_json', default='../dataset/mr_ali_split.json', help='Ruta al archivo JSON de splits de mr_ali')
    parser.add_argument('--output_h5', default='../dataset/combined_dataset.h5', help='Ruta de salida para el H5 combinado')
    parser.add_argument('--output_json', default='../dataset/combined_split.json', help='Ruta de salida para el JSON combinado')
    
    args = parser.parse_args()
    
    # Combinar archivos H5
    combine_h5_files(args.mr_hisum_h5, args.mr_ali_h5, args.output_h5)
    
    # Combinar archivos JSON
    combine_json_files(args.mr_hisum_json, args.mr_ali_json, args.output_json)
    
    print("¡Proceso completado!")
    print(f"Archivo H5 combinado guardado en: {args.output_h5}")
    print(f"Archivo JSON combinado guardado en: {args.output_json}")

if __name__ == "__main__":
    main() 