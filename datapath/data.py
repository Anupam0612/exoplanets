import os
import sys
from datafed.CommandLib import API
from datafed import version as df_ver

try:
    datapath = os.mkdir("./datapath")
except:
    datapath = "./datapath"

df_api = API()
print("Success! You have DataFed: "+ df_ver)

pl_resp = df_api.projectList()
print(pl_resp)

ls_resp = df_api.collectionItemsList("root", context="p/2023_mem680t")
print(ls_resp)

for record in ls_resp[0].item:
    print(record.id, "\t", record.alias)

import json
parameters = {
    "Title" : "Exoplanet Atmosphere Observability",
    "Description" : "The dataset used in this analysis contains information on confirmed exoplanets with various attributes, spanning across 18,428 different observations.",
    "Authors" : "Susan E. Mullally, David R. Rodriguez, Kevin B. Stevenson, Hannah R. Wakeford",
    "Affiliatons" : "Space Telescope Science Institute, Johns Hopkins Applied Physics Laboratory, Exoplanets.org, NExScI, TOI",
    "Date of Data Creation" : "December 2019",
    "Data Type" : "Numerical",
    "Format" : "CSV",
    "Keywords" : "Signal-to-Noise Ratio (SNR)",
    "Planet_Name" : {"Data Type" : "char", "Name, Units" : "Exoplanet Name"},
    "SNR_Emission_15_micron" : {"Data Type" : "double", "Name, Units" : "1.5 Micron Emission SNR relative to HD 209458 b"},
    "SNR_Emission_5_micron" : {"Data Type" : "double", "Name, Units" : "5 Micron Emission SNR relative to HD 209458 b"},
    "SNR_Transmission_K_mag" : {"Data Type" : "double", "Name, Units" : "K-band Transmission SNR relative to HD 209458 b"},
    "Rp" : {"Data Type" : "double", "Name, Units" : "Exoplanet Radius, Jupiter radii"},
    "Mp" : {"Data Type" : "double", "Name, Units" : "Exoplanet Mass, Jupiter Masses"},
    "Tday" : {"Data Type" : "double", "Name, Units" : "Exoplanet Dayside Temperature, Kelvin"},
    "Teq" : {"Data Type" : "double", "Name, Units" : "Exoplanet Equilibrium Temperature, Kelvin"},
    "log10g_p" : {"Data Type" : "double", "Name, Units" : "Exoplanet Surface Gravity, log(cm/s^2)"},
    "Period" : {"Data Type" : "double", "Name, Units" : "Exoplanet Orbital Period, Days"},
    "Transit_Duration" : {"Data Type" : "double", "Name, Units" : "Exoplanet Transit Duration, Hours"},
    "K_mag" : {"Data Type" : "double", "Name, Units" : "Infrared K-band Magnitude (Brightness), Mag"},
    "Distance" : {"Data Type" : "double", "Name, Units" : "Distance to Exoplanet Host Star, Parsecs"},
    "Teff" : {"Data Type" : "double", "Name, Units" : "Stellar Effective Temperature, Kelvin"},
    "log10g_s" : {"Data Type" : "double", "Name, Units" : "Stellar Surface Gravity, log(cm/s^2)"},
    "Transit_Flag" : {"Data Type" : "int", "Name, Units" : "Exoplanet Transit Flag (FALSE- no transit, TRUE- transit)"},
    "Catalog Name" : {"Data Type" : "char", "Name, Units" : "Exoplanet Source Cataog Name"}

}

parent_collection = "c/500594589"
dc_resp = df_api.dataCreate(
    "my important data",
    metadata=json.dumps(parameters),
    parent_id=parent_collection,
)
dc_resp

record_id = dc_resp[0].data[0].id
print(record_id)

put_resp = df_api.dataPut(
    record_id,
    "/Users/anupammishra/exoplanets/docs/search.csv",
    wait=True
)
print(put_resp)

# if df_api.getAuthUser():
#     print(
#         "Success! You have been authenticated into DataFed as: " + df_api.getAuthUser()
#     )
# else:
#     print("You have not authenticated into DataFed Client")
#     print(
#         'Please follow instructions in the "Basic Configuration" section in the link below to authenticate yourself:'
#     )
#     print("https://ornl.github.io/DataFed/user/client/install.html#basic-configuration")


# if not df_api.endpointDefaultGet():
#     print("Please follow instructions in the link below to find your Globus Endpoint:")
#     print(
#         "https://ornl.github.io/DataFed/system/getting_started.html#install-identify-globus-endpoint"
#     )
#     endpoint = input(
#         "\nPlease enter either the Endpoint UUID or Legacy Name for your Globus Endpoint: "
#     )
#     df_api.endpointDefaultSet(endpoint)

# print("Your default Globus Endpoint in DataFed is:\n" + df_api.endpointDefaultGet())


# # This is a dataGet Command
# dget_resp = df_api.dataGet("c/500019915", os.path.abspath(datapath), wait=True)
# dget_resp

# if dget_resp[0].task[0].status == 3:
#     print("Success! Downloaded a test file to your location. Removing the file now")
#     os.remove(os.path.abspath(datapath) + "/49965349.ibw")
# else:
#     if dget_resp[0].task[0].msg == "globus connect offline":
#         print(
#             "You need to activate your Globus Endpoint and/or ensure Globus Connect Personal is running.\n"
#             "Please visit https://globus.org to activate your Endpoint"
#         )
#     elif dget_resp[0].task[0].msg == "permission denied":
#         print(
#             "Globus does not have write access to this directory. \n"
#             "If you are using Globus Connect Personal, ensure that this notebook runs within"
#             "one of the directories where Globus has write access. You may consider moving this"
#             "notebook to a valid directory or add this directory to the Globus Connect Personal settings"
#         )
#     else:
#         NotImplementedError(
#             "Get in touch with us or consider looking online to find a solution to this problem:\n"
#             + dget_resp[0].task[0].msg
        # )
