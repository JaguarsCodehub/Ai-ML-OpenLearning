import pubchempy as pcp
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import os
from PIL import Image
import io
import imghdr

def download_and_verify_image(compound_id, output_dir):
    """
    Download and verify a single compound image
    """
    try:
        # Download image
        img_path = os.path.join(output_dir, f"compound_{compound_id}.png")
        if os.path.exists(img_path):
            # Verify existing image
            with Image.open(img_path) as img:
                img.verify()  # Verify it's a valid image
                return True
        
        # Download if not exists or invalid
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/{compound_id}/PNG"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(img_path, 'wb') as f:
                f.write(response.content)
            
            # Verify downloaded image
            with Image.open(img_path) as img:
                img.verify()
            return True
            
    except Exception as e:
        print(f"Error downloading/verifying compound {compound_id}: {str(e)}")
        if os.path.exists(img_path):
            os.remove(img_path)  # Remove corrupted file
        return False

def create_metadata(output_dir, successful_compounds):
    """Create metadata CSV file with compound information"""
    metadata = []
    print("\nCreating metadata:", flush=True)
    
    for compound_id in tqdm(successful_compounds, desc="Creating metadata"):
        try:
            # Get compound information from PubChem API
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/{compound_id}/property/MolecularFormula/JSON"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                formula = data['PropertyTable']['Properties'][0]['MolecularFormula']
                
                metadata.append({
                    'id': compound_id,
                    'formula': formula,
                    'image_path': f"compound_{compound_id}.png"
                })
            else:
                print(f"Failed to get formula for compound {compound_id}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"Error processing compound {compound_id}: {str(e)}")
            continue
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(metadata)
    csv_path = os.path.join(output_dir, 'metadata.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nMetadata saved to {csv_path} 📝")
    
    # Verify the data
    print("\nVerifying images...")
    valid_compounds = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Verifying images"):
        img_path = os.path.join(output_dir, row['image_path'])
        if os.path.exists(img_path) and imghdr.what(img_path) is not None:
            valid_compounds.append(row)
    
    # Update DataFrame with only valid compounds
    df_valid = pd.DataFrame(valid_compounds)
    df_valid.to_csv(csv_path, index=False)
    print(f"Found {len(df_valid)} valid compounds out of {len(df)} total")
    
    return df_valid

def download_pubchem_data(output_dir, n_compounds=100):
    """Download compound data with verification"""
    os.makedirs(output_dir, exist_ok=True)
    
    successful_compounds = []
    failed_compounds = []
    
    print(f"Downloading {n_compounds} compounds to {output_dir}")
    
    for compound_id in tqdm(range(1, n_compounds + 1), desc="Downloading compounds"):
        try:
            # Download image
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/{compound_id}/PNG"
            img_path = os.path.join(output_dir, f"compound_{compound_id}.png")
            
            response = requests.get(url)
            if response.status_code == 200:
                # Verify it's a valid image
                try:
                    img = Image.open(io.BytesIO(response.content))
                    img.verify()
                    
                    # Save the image
                    with open(img_path, 'wb') as f:
                        f.write(response.content)
                    
                    successful_compounds.append(compound_id)
                    
                except Exception as e:
                    print(f"Invalid image data for compound {compound_id}: {str(e)}")
                    failed_compounds.append(compound_id)
            else:
                print(f"Failed to download compound {compound_id}: HTTP {response.status_code}")
                failed_compounds.append(compound_id)
                
        except Exception as e:
            print(f"Error processing compound {compound_id}: {str(e)}")
            failed_compounds.append(compound_id)
            continue
    
    print(f"\nDownload complete!")
    print(f"Successfully downloaded: {len(successful_compounds)} compounds")
    print(f"Failed downloads: {len(failed_compounds)} compounds")
    
    # Create metadata file and return both values
    metadata_df = None
    if successful_compounds:
        metadata_df = create_metadata(output_dir, successful_compounds)
    else:
        print("No compounds were successfully downloaded!")
    
    return successful_compounds, metadata_df