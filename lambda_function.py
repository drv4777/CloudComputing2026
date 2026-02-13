from azure.storage.blob import BlobServiceClient

import pandas as pd

import io

import json

 

def process_nutritional_data_from_azurite():

    connect_str = (
    "DefaultEndpointsProtocol=http;"
    "AccountName=devstoreaccount1;"
    "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
    "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
)

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

 

    container_name = 'datasets'

    blob_name = 'All_Diets.csv'

 

    container_client = blob_service_client.get_container_client(container_name)

    blob_client = container_client.get_blob_client(blob_name)

 

    # Download blob content to bytes

    stream = blob_client.download_blob().readall()

    df = pd.read_csv(io.BytesIO(stream))

 

    # Calculate averages

    avg_macros = df.groupby('Diet_type')[['Protein(g)', 'Carbs(g)', 'Fat(g)']].mean()

 

    # Save results locally as JSON (simulate NoSQL storage)

    result = avg_macros.reset_index().to_dict(orient='records')

    with open('simulated_nosql/results.json', 'w') as f:

        json.dump(result, f)

 

    return "Data processed and stored successfully."

#Run the function
print(process_nutritional_data_from_azurite())
