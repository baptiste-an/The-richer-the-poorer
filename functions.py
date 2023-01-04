import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.reshape.merge import merge
from pandas.io.pytables import Fixed
import pymrio
import scipy.io
from matplotlib import colors as mcolors, rc_params
import seaborn as sns
import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.io as pio
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as mpatches
import matplotlib.markers as mmark
import matplotlib.lines as mlines
from scipy import stats
import country_converter as coco
from matplotlib.patches import ConnectionPatch
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
import os
import requests
import pyarrow.feather as feather

# ...function to download and save data from url......


def download(url: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split("/")[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))


# ........CREATE FUNCTION TO CONVERT ANY REGION NAME FORMAT TO A COMMON FORMAT................

dict_regions = dict()  # create a dict that will be used to rename regions
cc = coco.CountryConverter(
    include_obsolete=True
)  # documentation for coco here: https://github.com/konstantinstadler/country_converter
for i in [
    n for n in cc.valid_class if n != "name_short"
]:  # we convert all the regions in cc to name short and add it to the dict
    dict_regions.update(cc.get_correspondence_dict(i, "name_short"))
name_short = cc.ISO3as("name_short")["name_short"].values  # array containing all region names in short_name format


def dict_regions_update():
    """Adds to dict the encountered region names that were not in coco.

    If a region is wider than a country (for example "European Union"), it is added to "Z - Aggregated categories" in order to be deleted later.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    dict_regions["Bolivia (Plurinational State of)"] = "Bolivia"
    dict_regions["Czechia"] = "Czech Republic"
    dict_regions["Iran (Islamic Republic of)"] = "Iran"
    dict_regions["China, Taiwan Province of China"] = "Taiwan"
    dict_regions["Congo"] = "Congo Republic"
    dict_regions["Venezuela (Bolivarian Republic of)"] = "Venezuela"
    dict_regions["Dem. People's Republic of Korea"] = "North Korea"
    dict_regions["Bahamas, The"] = "Bahamas"
    dict_regions["Congo, Dem. Rep."] = "DR Congo"
    dict_regions["Congo, Rep."] = "Congo Republic"
    dict_regions["Egypt, Arab Rep."] = "Egypt"
    dict_regions["Faroe Islands"] = "Faeroe Islands"
    dict_regions["Gambia, The"] = "Gambia"
    dict_regions["Hong Kong SAR, China"] = "Hong Kong"
    dict_regions["Iran, Islamic Rep."] = "Iran"
    dict_regions["Korea, Dem. People's Rep."] = "North Korea"
    dict_regions["Korea, Rep."] = "South Korea"
    dict_regions["Lao PDR"] = "Laos"
    dict_regions["Macao SAR, China"] = "Macau"
    dict_regions["North Macedonia"] = "Macedonia"
    dict_regions["Russian Federation"] = "Russia"
    dict_regions["Sint Maarten (Dutch part)"] = "Sint Maarten"
    dict_regions["Slovak Republic"] = "Slovakia"
    dict_regions["St. Martin (French part)"] = "Saint-Martin"
    dict_regions["Syrian Arab Republic"] = "Syria"
    dict_regions["Virgin Islands (U.S.)"] = "United States Virgin Islands"
    dict_regions["West Bank and Gaza"] = "Palestine"
    dict_regions["Yemen, Rep."] = "Yemen"
    dict_regions["Venezuela, RB"] = "Venezuela"
    dict_regions["Brunei"] = "Brunei Darussalam"
    dict_regions["Cape Verde"] = "Cabo Verde"
    dict_regions["Dem. People's Rep. Korea"] = "North Korea"
    dict_regions["Swaziland"] = "Eswatini"
    dict_regions["Taiwan, China"] = "Taiwan"
    dict_regions["Virgin Islands"] = "United States Virgin Islands"
    dict_regions["Yemen, PDR"] = "Yemen"
    dict_regions["Réunion"] = "Reunion"
    dict_regions["Saint Helena"] = "St. Helena"
    dict_regions["China, Hong Kong SAR"] = "Hong Kong"
    dict_regions["China, Macao SAR"] = "Macau"
    dict_regions["Bonaire, Sint Eustatius and Saba"] = "Bonaire, Saint Eustatius and Saba"
    dict_regions["Curaçao"] = "Curacao"
    dict_regions["Saint Barthélemy"] = "St. Barths"
    dict_regions["Saint Martin (French part)"] = "Saint-Martin"
    dict_regions["Micronesia (Fed. States of)"] = "Micronesia, Fed. Sts."
    dict_regions["Micronesia, Federated State=s of"] = "Micronesia, Fed. Sts."
    dict_regions["Bonaire"] = "Bonaire, Saint Eustatius and Saba"
    dict_regions["São Tomé and Principe"] = "Sao Tome and Principe"
    dict_regions["Virgin Islands, British"] = "British Virgin Islands"
    dict_regions["Wallis and Futuna"] = "Wallis and Futuna Islands"
    dict_regions["Micronesia, Federated States of"] = "Micronesia, Fed. Sts."

    for j in [
        "Africa Eastern and Southern",
        "Africa Western and Central",
        "Arab World",
        "Caribbean small states",
        "Central Europe and the Baltics",
        "Early-demographic dividend",
        "East Asia & Pacific",
        "East Asia & Pacific (excluding high income)",
        "East Asia & Pacific (IDA & IBRD countries)",
        "Euro area",
        "Europe & Central Asia",
        "Europe & Central Asia (excluding high income)",
        "Europe & Central Asia (IDA & IBRD countries)",
        "European Union",
        "Fragile and conflict affected situations",
        "Heavily indebted poor countries (HIPC)",
        "High income",
        "IBRD only",
        "IDA & IBRD total",
        "IDA blend",
        "IDA only",
        "IDA total",
        "Late-demographic dividend",
        "Latin America & Caribbean",
        "Latin America & Caribbean (excluding high income)",
        "Latin America & the Caribbean (IDA & IBRD countries)",
        "Least developed countries: UN classification",
        "Low & middle income",
        "Low income",
        "Lower middle income",
        "Middle East & North Africa",
        "Middle East & North Africa (excluding high income)",
        "Middle East & North Africa (IDA & IBRD countries)",
        "Middle income",
        "North America",
        "Not classified",
        "OECD members",
        "Other small states",
        "Pacific island small states",
        "Post-demographic dividend",
        "Pre-demographic dividend",
        "Small states",
        "South Asia",
        "South Asia (IDA & IBRD)",
        "Sub-Saharan Africa",
        "Sub-Saharan Africa (excluding high income)",
        "Sub-Saharan Africa (IDA & IBRD countries)",
        "Upper middle income",
        "World",
        "Arab League states",
        "China and India",
        "Czechoslovakia",
        "East Asia & Pacific (all income levels)",
        "East Asia & Pacific (IDA & IBRD)",
        "East Asia and the Pacific (IFC classification)",
        "EASTERN EUROPE",
        "Europe & Central Asia (all income levels)",
        "Europe & Central Asia (IDA & IBRD)",
        "Europe and Central Asia (IFC classification)",
        "European Community",
        "High income: nonOECD",
        "High income: OECD",
        "Latin America & Caribbean (all income levels)",
        "Latin America & Caribbean (IDA & IBRD)",
        "Latin America and the Caribbean (IFC classification)",
        "Low income, excluding China and India",
        "Low-income Africa",
        "Middle East & North Africa (all income levels)",
        "Middle East & North Africa (IDA & IBRD)",
        "Middle East (developing only)",
        "Middle East and North Africa (IFC classification)",
        "Other low-income",
        "Serbia and Montenegro",
        "Severely Indebted",
        "South Asia (IFC classification)",
        "Sub-Saharan Africa (all income levels)",
        "SUB-SAHARAN AFRICA (excl. Nigeria)",
        "Sub-Saharan Africa (IDA & IBRD)",
        "Sub-Saharan Africa (IFC classification)",
        "WORLD",
        "UN development groups",
        "More developed regions",
        "Less developed regions",
        "Least developed countries",
        "Less developed regions, excluding least developed countries",
        "Less developed regions, excluding China",
        "Land-locked Developing Countries (LLDC)",
        "Small Island Developing States (SIDS)",
        "World Bank income groups",
        "High-income countries",
        "Middle-income countries",
        "Upper-middle-income countries",
        "Lower-middle-income countries",
        "Low-income countries",
        "No income group available",
        "Geographic regions",
        "Latin America and the Caribbean",
        "Sustainable Development Goal (SDG) regions",
        "SUB-SAHARAN AFRICA",
        "NORTHERN AFRICA AND WESTERN ASIA",
        "CENTRAL AND SOUTHERN ASIA",
        "EASTERN AND SOUTH-EASTERN ASIA",
        "LATIN AMERICA AND THE CARIBBEAN",
        "AUSTRALIA/NEW ZEALAND",
        "OCEANIA (EXCLUDING AUSTRALIA AND NEW ZEALAND)",
        "EUROPE AND NORTHERN AMERICA",
        "EUROPE",
        "Holy See",
        "NORTHERN AMERICA",
        "East Asia & Pacific (ICP)",
        "Europe & Central Asia (ICP)",
        "Latin America & Caribbean (ICP)",
        "Middle East & North Africa (ICP)",
        "North America (ICP)",
        "South Asia (ICP)",
        "Sub-Saharan Africa (ICP)",
    ]:
        dict_regions[j] = "Z - Aggregated categories"
    return None


dict_regions_update()
# all the regions that do not correspond to a country are in 'Z - Aggregated categories'
# rename the appropriate level of dataframe using dict_regions
def rename_region(df, level="LOCATION"):
    """Renames the regions of a DataFrame into name_short format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose regions must be renamed
    level : string
        Name of the level containing the region names

    Returns
    df : pd.DataFrame
        DataFrame with regions in name_short format
    -------
    None
    """
    if level in df.index.names:
        axis = 0
    else:
        axis = 1
        df = df.T

    index_names = df.index.names
    df = df.reset_index()
    df = df.set_index(level)
    df = df.rename(index=dict_regions)  # rename index according to dict
    ind = df.index.values
    for i in range(0, len(ind), 1):
        if type(ind[i]) == list:
            # if len(ind[i])==0:
            ind[i] = ind[i][0]
    df = df.reindex(ind)
    df = df.reset_index().set_index(index_names)
    for i in df.index.get_level_values(level).unique():
        if i not in name_short and i != i == "Z - Aggregated categories":
            print(
                i
                + " is not in dict_regions\nAdd it using\n  >>> dict_regions['"
                + i
                + "'] = 'region' # name_short format\n"
            )
    if axis == 1:
        df = df.T
    return df


# ........CREATE KBAR MATRIX FOR YEAR 2017................


def Kbar():
    """Calculates Kbar for year 2017 from Kbar2015, Kbar2014, CFC2017 and GFCF2017.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    CFC_WB = pd.read_excel("World Bank/CFC worldbank.xlsx", header=3, index_col=[0])["2017"]

    GFCF_WB = pd.read_excel("World Bank/GFCF worldbank.xlsx", header=3, index_col=[0])["2017"]

    # we want to set to NaN values for regions that don't have both CFC and GFCF data
    CFC_WB = CFC_WB * GFCF_WB / GFCF_WB  # in case some countries have data for CFC but not GFCF
    GFCF_WB = GFCF_WB * CFC_WB / CFC_WB

    CFC_WB = rename_region(CFC_WB, "Country Name").drop("Z - Aggregated categories")
    # rename the regions to a common format
    CFC_WB["region"] = cc.convert(names=CFC_WB.index, to="EXIO3")
    # convert the common format to EXIOBASE format
    CFC_WB = CFC_WB.reset_index().set_index("region").drop("Country Name", axis=1).groupby(level="region").sum()
    # define EXIOBASE regions as index

    GFCF_WB = rename_region(GFCF_WB, "Country Name").drop("Z - Aggregated categories")
    GFCF_WB["region"] = cc.convert(names=GFCF_WB.index, to="EXIO3")
    GFCF_WB = GFCF_WB.reset_index().set_index("region").drop("Country Name", axis=1).groupby(level="region").sum()

    GFCF_over_CFC_WB = GFCF_WB["2017"] / CFC_WB["2017"]
    GFCF_over_CFC_WB.loc["TW"] = GFCF_over_CFC_WB.loc["CN"]
    # hypothesis: ratio of GFCF/CFC is same for Taiwan than for China

    Z = pd.read_csv(
        "Data/EXIO3/IOT_2017_pxp/Z.txt",
        delimiter="\t",
        header=[0, 1],
        index_col=[0, 1],
    )  # Z read from exiobase data
    Zcoefs = Z.div(Z.sum(axis=1), axis=0)

    mat15 = scipy.io.loadmat("Data/Sodersten/Kbar_exio_v3_6_2015pxp")
    Kbar15 = pd.DataFrame(mat15["KbarCfc"].toarray(), index=Z.index, columns=Z.columns)

    # We calculate coefficients for year 2015 that will be multiplied by CFC for year 2017
    Kbarcoefs = Kbar15.div(Kbar15.sum(axis=1), axis=0)  # the sum of the lines = CFC

    Kbarcoefs = Kbarcoefs.combine_first(Zcoefs).combine_first(
        pd.DataFrame(np.identity(9800), index=Z.index, columns=Z.columns)
    )  # if some coefs don't exist in Kbar15, we use those of Z

    GFCF_exio = (
        pd.read_csv(
            "Data/EXIO3/IOT_2017_pxp/Y.txt",
            delimiter="\t",
            header=[0, 1],
            index_col=[0, 1],
        )
        .swaplevel(axis=1)["Gross fixed capital formation"]
        .sum()
    )  # aggregated 49 regions, 1 product
    CFC_exio = pd.read_csv(
        "Data/EXIO3/IOT_2017_pxp/satellite/F.txt",
        delimiter="\t",
        header=[0, 1],
        index_col=[0],
    ).loc[
        "Operating surplus: Consumption of fixed capital"
    ]  # 49 regions 200 sectors
    GFCF_over_CFC_exio = GFCF_exio / CFC_exio.unstack().sum(axis=1)  # 49 regions 1 sector

    # we rescale CFC in order to obtain ratio GFCF/CFC of the worldbank when all sectors are aggregated
    CFC_rescaled = (
        CFC_exio.unstack()  # 49 regions 200 sectors
        .mul(GFCF_over_CFC_exio, axis=0)  # 49 regions 1 sector
        .div(GFCF_over_CFC_WB, axis=0)  # 49 regions 1 sector
        .stack()
    )

    feather.write_feather(Z, "Data/EXIO3/IOT_2017_pxp/Z.feather")
    # files .txt take too long to read, feather files save time for functions Y_all() and Lk()

    feather.write_feather(Kbarcoefs.mul(CFC_rescaled, axis=0), "Results/Kbar_2017pxp.feather")


# ........CREATE FUNCTION TO AGGREGATE A DATAFRAME FROM GIVEN CONCORDANCE TABLE................


def agg(df: pd.DataFrame, table: pd.DataFrame, axis=0) -> pd.DataFrame:
    """Aggregates a DataFrame on the level specified in the concordance table, on the axis specified.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be aggregated
    table : pd.DataFrame
        Concordance table for the agregation.
        It can be one-to-one or one-to-many but be of the appropriate shape.
        If one-to-one index_col=[0,1]. If one to many, index_col=0 and header=0.
        The name of the level to be aggregated must be the name of index_col=0.
    axis : int
        Axis that contains the level to be aggregated. Default=0.

    Returns
    -------
    df_agg : pd.DataFrame
        The aggregated DataFrame
    """
    if isinstance(table.index, pd.MultiIndex):
        if axis == 0:
            df_agg = df.rename(index=dict(table.index)).groupby(level=df.index.names).sum()
        else:
            df_agg = df.rename(columns=dict(table.index)).groupby(level=df.columns.names, axis=1).sum()
        return df_agg

    else:
        if axis == 1:
            df = df.T

        levels_to_unstack = list(df.index.names)
        levels_to_unstack.remove(table.index.names[0])
        df = df.unstack(levels_to_unstack)

        df_agg = pd.DataFrame()
        for i in table.columns:
            df_agg[i] = df.mul(table[i], axis=0).sum()
        df_agg.columns.names = [table.index.names[0]]

        if axis == 0:
            return df_agg.unstack(levels_to_unstack).T
        else:
            return df_agg.unstack(levels_to_unstack)


# .........CALCULATIONS......................................

# function to diaggregate GDP into all the formats needed
def Y_all():

    """Adds Ytot, Yh, Yg, Yr and Ygfcf to the raw Exiobase data files.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    Z = feather.read_feather(
        "Data/EXIO3/IOT_2017_pxp/Z.feather"
    )  # we only use it to access the index and columnsin order to get index and columns for Kbar

    pathIOT = "Data/EXIO3/IOT_2017_pxp/"
    Y = pd.read_csv(
        pathIOT + "Y.txt",
        delimiter="\t",
        header=[0, 1],
        index_col=[0, 1],
    )

    # total GDP
    Ytot = Y.groupby(level="region", axis=1).sum()

    # households and NPISH final consumption
    Yh = (
        Y.swaplevel(axis=1)[
            [
                "Final consumption expenditure by households",
                "Final consumption expenditure by non-profit organisations serving households (NPISH)",
            ]
        ]
        .groupby(level="region", axis=1)
        .sum()
    )

    # government final consumption
    Yg = Y.swaplevel(axis=1)["Final consumption expenditure by government"].groupby(level="region", axis=1).sum()

    # other components of GDP
    Yother = (
        Y.swaplevel(axis=1)[
            [
                "Changes in inventories",
                "Changes in valuables",
                "Exports: Total (fob)",
            ]
        ]
        .groupby(level="region", axis=1)
        .sum()
    )

    Kbar = feather.read_feather("Results/Kbar_2017pxp.feather")
    Kbar = pd.DataFrame(Kbar, index=Z.index, columns=Z.columns).fillna(0)

    NET = pd.DataFrame(index=Y.index)
    CFC = pd.DataFrame(index=Y.index)

    for region in Y.stack().columns:

        cfc = Kbar.loc[region].sum(axis=1)  # CFC of region for each one of the 200 sectors

        # where does the CFC come from? We use the same shares as GCF data

        # à comparer avec GCF du pays
        gcf_all = (
            Y.stack(level=0)[
                [
                    "Changes in inventories",
                    "Changes in valuables",
                    "Exports: Total (fob)",
                    "Gross fixed capital formation",
                ]
            ]
            .sum(axis=1)
            .unstack()[region]
            .unstack()
        )  # lines=regions, columns=sectors of GCF

        gcf_shares = gcf_all.div(gcf_all.sum(), axis=1)  # share of origin region for each sector
        gcf_shares.loc[region, gcf_shares.isnull().all()] = 1
        gcf = gcf_all.sum()
        diff = gcf - cfc
        NET[region] = gcf_shares.mul(diff, axis=1).stack()
        CFC[region] = gcf_shares.mul(cfc, axis=1).stack()

    # df["CFC"] = CFC.stack()
    # df["NCF"] = NET.stack()
    Yr = NET # residual Y (net formation of capital)
    # Yr = Ytot - Yg - Yh - Yother - Kbar.groupby(level="region", axis=1).sum()

    Ygfcf = Y.swaplevel(axis=1)["Gross fixed capital formation"]

    feather.write_feather(Ytot, "Results/Ytot.feather")
    feather.write_feather(Yh, "Results/Yh.feather")
    feather.write_feather(Yg, "Results/Yg.feather")
    feather.write_feather(Yr, "Results/Yr.feather")
    feather.write_feather(Yother, "Results/Yother.feather")
    feather.write_feather(Ygfcf, "Results/Ygfcf.feather")

    return "All Y files were saved in Results"


# function to calculate Lk from Z and Kbar
def Lk():
    """Calculates Lk from Kbar and saves it to the Exiobase data files.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    Y = (
        pd.read_csv(
            "Data/EXIO3/IOT_2017_pxp/Y.txt",
            delimiter="\t",
            header=[0, 1],
            index_col=[0, 1],
        )
        .groupby(level="region", axis=1)
        .sum()
    )

    Z = feather.read_feather("Data/EXIO3/IOT_2017_pxp/Z.feather")

    Kbar = feather.read_feather("Results/Kbar_2017pxp.feather")
    Kbar = pd.DataFrame(Kbar, index=Z.index, columns=Z.columns).fillna(0)

    Zk = Z + Kbar
    x = Z.sum(axis=1) + Y.sum(axis=1)
    Ak = pymrio.calc_A(Zk, x)
    A = pymrio.calc_A(Z, x)

    feather.write_feather(pymrio.calc_L(Ak), "Results/Lk.feather")
    feather.write_feather(pymrio.calc_L(A), "Data/EXIO3/IOT_2017_pxp/L.feather")

    return "Lk saved in Results"


# function to calculate output associated with L and Lk for each component of Y
def LY():
    """Calculates LY with all the possible combinations of L and Y.

    Parameters
    ----------

    Returns
    -------
    None
    """

    conc = pd.read_excel("concordance.xlsx", sheet_name="final consumption", index_col=0)
    conc.index.names = ["sector cons"]

    Yh = feather.read_feather("Results/Yh.feather")
    Yg = feather.read_feather("Results/Yg.feather")
    Yr = feather.read_feather("Results/Yr.feather")
    Ygfcf = feather.read_feather("Results/Ygfcf.feather")
    Yother = feather.read_feather("Results/Yother.feather")

    L = feather.read_feather("Data/EXIO3/IOT_2017_pxp/L.feather")
    L.columns.names = ["region cons", "sector cons"]
    L.index.names = ["region prod", "sector prod"]
    Lk = feather.read_feather("Results/Lk.feather")
    Lk.columns.names = ["region cons", "sector cons"]
    Lk.index.names = ["region prod", "sector prod"]

    LY_all = pd.DataFrame()
    l = 0
    for Y in [Yh, Yg, Yother]:
        Y.columns.names = ["region cons"]
        Y.index.names = ["region prod", "sector prod"]
        LY = pd.DataFrame()
        LkY = pd.DataFrame()
        for j in Y.columns:
            LY[j] = agg(
                L.mul(Y[j], axis=1).groupby(level="sector cons", axis=1).sum(),
                conc,
                axis=1,
            ).stack()
            LkY[j] = agg(
                Lk.mul(Y[j], axis=1).groupby(level="sector cons", axis=1).sum(),
                conc,
                axis=1,
            ).stack()
        columns_L = ["LYh", "LYg", "LYother"]
        columns_Lk = ["LkYh", "LkYg", "LkYother"]
        LY_all[columns_L[l]] = LY.stack()
        LY_all[columns_Lk[l]] = LkY.stack()
        l += 1

    conc_cap = pd.read_excel("concordance.xlsx", sheet_name="capital", index_col=[0, 1])
    conc_cap.index.names = ["sector cons", "ICP sector"]
    LY_gfcf = pd.DataFrame()
    LYgfcf = pd.DataFrame()
    LkYr = pd.DataFrame()
    Ygfcf.columns.names = ["region cons"]
    Ygfcf.index.names = ["region prod", "sector prod"]
    for j in Ygfcf.columns:
        LkYr[j] = agg(
            Lk.mul(Yr[j], axis=1).groupby(level="sector cons", axis=1).sum(),
            conc_cap,
            axis=1,
        ).stack()
        LYgfcf[j] = agg(
            L.mul(Ygfcf[j], axis=1).groupby(level="sector cons", axis=1).sum(),
            conc_cap,
            axis=1,
        ).stack()

    LY_gfcf = pd.concat([LYgfcf.stack(), LkYr.stack()], axis=1, keys=["LYgfcf", "LkYr"])

    LY = pd.concat(
        [LY_all.unstack(level="sector cons"), LY_gfcf.unstack(level="sector cons")],
        axis=1,
    )

    feather.write_feather(LY.stack(), "Results/LY.feather")

    return "LYh, LkYh, LYg etc. were saved in Results"


# associates energy extensions to LY
def SLY():

    """Calculates SLY with all the possible combinations of L,Y and a set of impacts and satellites.
    Production sectors are aggregated according to the table given in concordance_path.
    49 means that regions have not been aggregated and the matrix is still of dimenssion 49reg*49reg.

    Parameters
    ----------

    Returns
    -------
    None
    """

    SLY = pd.DataFrame()

    conc_sec_prod = pd.read_excel(
        "concordance.xlsx",
        sheet_name="final consumption",
        index_col=[0],
    )
    conc_sec_prod.index.names = ["sector prod"]

    LY = feather.read_feather("Results/LY.feather").unstack().unstack()
    LY.columns.names = ["LY name", "sector cons", "region cons"]
    LY.index.names = ["region prod", "sector prod"]

    S = pd.read_csv(
        "Data/EXIO3/IOT_2017_pxp/satellite/S.txt",
        delimiter="\t",
        header=[0, 1],
        index_col=[0],
    )
    S.columns.names = ["region prod", "sector prod"]

    S_imp = pd.read_csv(
        "Data/EXIO3/IOT_2017_pxp/impacts/S.txt",
        delimiter="\t",
        header=[0, 1],
        index_col=[0],
    )
    S_imp.columns.names = ["region prod", "sector prod"]

    SLY["GHG"] = (
        agg(
            LY.mul(
                S_imp.loc[
                    "GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"
                ],
                axis=0,
            ),
            conc_sec_prod,
        )
        .unstack()
        .unstack()
    )

    SLY["Energy Carrier Net Total"] = (
        agg(LY.mul(S.loc["Energy Carrier Net Total"], axis=0), conc_sec_prod).unstack().unstack()
    )
    SLY["Energy Carrier Net LOSS"] = (
        agg(LY.mul(S.loc["Energy Carrier Net LOSS"], axis=0), conc_sec_prod).unstack().unstack()
    )

    SLY.columns.names = ["Extensions"]
    feather.write_feather(SLY, "Results/SLY.feather")

    return "SLYh, SLkYh, etc. were saved in Results"


# ..............ICP CALCULATION..............

sect_cap = [
    "150100:MACHINERY AND EQUIPMENT",
    "150200:CONSTRUCTION",
    "150300:OTHER PRODUCTS",
]  # names of GFCF disaggregation


def ppp_calculations():

    # read data from International Comparison Panel
    ICP2017 = (
        pd.read_excel(
            "ICP/Data_Extract_From_ICP_2017.xlsx",
            index_col=[0, 2, 4],
        )[:-5]
        .drop(["Country Code", "Classification Code", "Series Code"], axis=1)["2017 [YR2017]"]
        .unstack(level="Classification Name")
    ).replace({"..": np.nan})
    sector_dict = dict(
        {
            "1101000:FOOD AND NON-ALCOHOLIC BEVERAGES": "CPI: 01 - Food and non-Alcoholic beverages",
            "1102000:ALCOHOLIC BEVERAGES, TOBACCO AND NARCOTICS": "CPI: 02 - Alcoholic beverages, tobacco and narcotics",
            "1103000:CLOTHING AND FOOTWEAR": "CPI: 03 - Clothing and footwear",
            "9060000:ACTUAL HOUSING, WATER, ELECTRICITY, GAS AND OTHER FUELS": "CPI: 04 - Housing, water, electricity, gas and other fuels",
            "1105000:FURNISHINGS, HOUSEHOLD EQUIPMENT AND ROUTINE HOUSEHOLD MAINTENANCE": "CPI: 05 - Furnishings, household equipment and routine household maintenance",
            "9080000:ACTUAL HEALTH": "CPI: 06 - Health",
            "1107000:TRANSPORT": "CPI: 07 - Transport",
            "1108000:COMMUNICATION": "CPI: 08 - Communication",
            "9110000:ACTUAL RECREATION AND CULTURE": "CPI: 09 - Recreation and culture",
            "9120000:ACTUAL EDUCATION": "CPI: 10 - Education",
            "1111000:RESTAURANTS AND HOTELS": "CPI: 11 - Restaurants and hotels",
            "9140000:ACTUAL MISCELLANEOUS GOODS AND SERVICES": "CPI: 12 - Miscellaneous goods and services",
            "9270000:GENERAL GOVERNMENT FINAL CONSUMPTION EXPENDITURE": "Final Consumption",
            "1501100:MACHINERY AND EQUIPMENT": "150100:MACHINERY AND EQUIPMENT",
            "1501200:CONSTRUCTION": "150200:CONSTRUCTION",
            "1501300:OTHER PRODUCTS": "150300:OTHER PRODUCTS",
        }
    )

    # real GDP from ICP
    real = ICP2017["Expenditure, PPP-based (US$, billions)"].unstack(level="Series Name").rename(columns=sector_dict)
    # aggregation into 49 EXIOBASE regions
    real_agg = rename_region(real, "Country Name").drop("Z - Aggregated categories")
    real_agg["region"] = cc.convert(names=real_agg.index, to="EXIO3")
    real_agg = real_agg.reset_index().set_index(["Country Name", "region"]).groupby(level="region").sum()

    # nominal GDP from ICP
    nominal = (
        ICP2017["Expenditure, market exchange rate-based (US$, billions)"]
        .unstack(level="Series Name")
        .rename(columns=sector_dict)
    )
    # aggregation into 49 EXIOBASE regions
    nominal_agg = rename_region(nominal, "Country Name").drop("Z - Aggregated categories")
    nominal_agg["region"] = cc.convert(names=nominal_agg.index, to="EXIO3")
    nominal_agg = nominal_agg.reset_index().set_index(["Country Name", "region"]).groupby(level="region").sum()

    # relative price index, calculated to use it on EXIOBASE economic data
    index = nominal_agg / real_agg

    # exchange rates of euro area from OECD database to transform EXIOBASE data in current dollars
    XReuros = pd.read_excel("OECD/exchange rates euro area.xlsx").loc[0]

    # read EXIOBASE data
    pathIOT = "Data/EXIO3/IOT_2017_pxp/"
    Y = pd.read_csv(
        pathIOT + "Y.txt",
        delimiter="\t",
        header=[0, 1],
        index_col=[0, 1],
    )
    Yh = (
        Y.swaplevel(axis=1)[
            [
                "Final consumption expenditure by households",
                "Final consumption expenditure by non-profit organisations serving households (NPISH)",
            ]
        ]
        .groupby(level="region", axis=1)
        .sum()
    )
    Yg = Y.swaplevel(axis=1)["Final consumption expenditure by government"].groupby(level="region", axis=1).sum()
    Ygfcf = Y.swaplevel(axis=1)["Gross fixed capital formation"]

    # EXIOBASE data from euros to dollars
    Yh_dollars = Yh / XReuros.loc[2017]
    Yg_dollars = Yg / XReuros.loc[2017]
    Ygfcf_dollars = Ygfcf / XReuros.loc[2017]

    # Aggregation of data to COICOP sectors
    conc = pd.read_excel("concordance.xlsx", sheet_name="final consumption", index_col=[0])
    conc_cap = pd.read_excel("concordance.xlsx", sheet_name="capital", index_col=[0, 1])
    Yh_dollars_coicop = agg(Yh_dollars.groupby(level="sector").sum(), conc)
    Yg_dollars_coicop = agg(Yg_dollars.groupby(level="sector").sum(), conc)
    Ygfcf_dollars_agg = agg(Ygfcf_dollars.groupby(level="sector").sum(), conc_cap)

    #
    Y_nominal = pd.concat([Yh_dollars_coicop + Yg_dollars_coicop, Ygfcf_dollars_agg])
    Y_real = Y_nominal / index.T.loc[Y_nominal.index]

    pop = ICP2017["Population"].unstack(level="Series Name")
    pop_agg = rename_region(pop, "Country Name").drop("Z - Aggregated categories")
    pop_agg["region"] = cc.convert(names=pop_agg.index, to="EXIO3")
    pop_agg = (
        pop_agg.reset_index()
        .set_index(["Country Name", "region"])
        .groupby(level="region")
        .sum()["SP.POP.TOTL.ICP:Population"]
    )

    real_gdp_cap = real_agg["1000000:GROSS DOMESTIC PRODUCT"] / pop_agg * 1000000000
    fc_cap = Y_real.drop(sect_cap).sum() / pop_agg * 1000000

    return (
        real_gdp_cap,
        real_agg,
        pop_agg,
        Y_real,
        fc_cap,
        index,
        Yh_dollars,
        Yg_dollars,
        Yh,
        Yg,
    )


# real_gdp_cap is the GDP for 49 regions in US$ppp
# ICP_data_real is the raw ICP data aggregated into 49 regions
# pop_agg is the population aggregated into 49 regions
# Y_real is the final consumption of households and governemnts (12 sectors) plus GFCF (3 sectors) for 49 regions in US$2017ppp
# fc_cap is the final consumption per capita in US$2017ppp/cap


# .........ENERGY INTENSITIES.............


# calculate all the energy vectors used to calculate energy intensities
def energy():
    SLY = feather.read_feather("Results/SLY.feather")
    SLY_pri = SLY["Energy Carrier Net Total"].unstack(level=0)
    SLY_fin = (SLY["Energy Carrier Net Total"] - SLY["Energy Carrier Net LOSS"]).unstack(level=0)
    E_K_fc_fin = (
        SLY_fin[["LkYh", "LkYg"]].sum(axis=1).unstack(level=["region cons"]).groupby(level="sector cons").sum()
    ).drop(sect_cap)
    E_K_fc_pri = (
        SLY_pri[["LkYh", "LkYg"]].sum(axis=1).unstack(level=["region cons"]).groupby(level="sector cons").sum()
    ).drop(sect_cap)

    F_hh = pd.read_csv(
        "Data/EXIO3/IOT_2017_pxp/satellite/F_Y.txt",
        delimiter="\t",
        header=[0, 1],
        index_col=[0],
    )
    F_hh.columns.names = ["region prod", "sector prod"]
    E_hhpri = F_hh.loc["Energy Carrier Net Total"].groupby(level="region prod").sum()
    E_hhfin = (
        (F_hh.loc["Energy Carrier Net Total"] - F_hh.loc["Energy Carrier Net LOSS"]).groupby(level="region prod").sum()
    )

    D_pba = pd.read_csv(
        "Data/EXIO3/IOT_2017_pxp/satellite/D_pba.txt",
        delimiter="\t",
        header=[0, 1],
        index_col=[0],
    )
    D_pba.columns.names = ["region prod", "sector prod"]
    E_pbapri = D_pba.loc["Energy Carrier Net Total"].groupby(level="region prod").sum()
    E_pbafin = (
        (D_pba.loc["Energy Carrier Net Total"] - D_pba.loc["Energy Carrier Net LOSS"])
        .groupby(level="region prod")
        .sum()
    )

    D_cba = pd.read_csv(
        "Data/EXIO3/IOT_2017_pxp/satellite/D_cba.txt",
        delimiter="\t",
        header=[0, 1],
        index_col=[0],
    )
    D_cba.columns.names = ["region prod", "sector prod"]
    E_cbapri = D_cba.loc["Energy Carrier Net Total"].groupby(level="region prod").sum()
    E_cbafin = (
        (D_cba.loc["Energy Carrier Net Total"] - D_cba.loc["Energy Carrier Net LOSS"])
        .groupby(level="region prod")
        .sum()
    )

    return (
        E_K_fc_fin,
        E_K_fc_pri,
        E_hhfin,
        E_hhpri,
        E_pbafin,
        E_pbapri,
        E_cbafin,
        E_cbapri,
    )


# ..................VALIDATION......................

# check that the energy footprint is the same with and without capital endogenized
def validation():
    SLY = feather.read_feather("Results/SLY.feather")
    a = (
        SLY.sum(axis=1).unstack(level="LY name")[["LYg", "LYgfcf", "LYh", "LYother"]].sum().sum()
        / SLY.sum(axis=1).unstack(level="LY name")[["LkYg", "LkYh", "LkYother", "LkYr"]].sum().sum()
    )
    return a - 1


# ..........DATA FOR FIGURES...................
def data_figure1():

    nominal = (
        pd.read_excel("ICP/Data_Extract_From_ICP_2017.xlsx", index_col=[0, 2, 4],)[:-5][
            "2017 [YR2017]"
        ].unstack(level="Classification Name")
    )["Expenditure, market exchange rate-based (US$, billions)"].loc["WORLD"]
    a = nominal.loc["1501000:GROSS FIXED CAPITAL FORMATION"]
    b = nominal.loc[
        [
            "1502000:CHANGES IN INVENTORIES",
            "1503000:ACQUISITIONS LESS DISPOSALS OF VALUABLES (Category)",
            "1600000:BALANCE OF EXPORTS AND IMPORTS",
        ]
    ].sum()
    c = nominal.loc[
        [
            "1300000:INDIVIDUAL CONSUMPTION EXPENDITURE BY GOVERNMENT",
            "1400000:COLLECTIVE CONSUMPTION EXPENDITURE BY GOVERNMENT",
            "9100000:HOUSEHOLDS AND NPISHS FINAL CONSUMPTION EXPENDITURE",
        ]
    ].sum()

    CFC = (
        rename_region(
            pd.read_excel("World Bank/CFC worldbank.xlsx", header=3, index_col=[0])["2017"],
            "Country Name",
        )
        .drop("Z - Aggregated categories")
        .sum()
    )
    GFCF = (
        rename_region(
            pd.read_excel("World Bank/GFCF worldbank.xlsx", header=3, index_col=[0])["2017"],
            "Country Name",
        )
        .drop("Z - Aggregated categories")
        .sum()
    )
    CFC_share = (CFC / GFCF).loc["2017"]

    pd.DataFrame(
        [
            [a / (a + b + c), b / (a + b + c), c / (a + b + c), 0, 0],
            [0, 0, 0, 1 - CFC_share, CFC_share],
        ],
        columns=[
            "Gross Fixed Capital\nFormation",
            "Rest of Gross\nCapital Formation",
            "Final consumption\nexpenditure",
            "Net Value\nAdded",
            "Consumption of\nFixed Capital",
        ],
        index=["pie1", "pie2"],
    ).T.to_excel("Figures/figure1.xlsx")


def data_figure2(E_hhfin, pop_agg, E_K_fc_fin, E_pbafin, E_cbafin, real_gdp_cap, fc_cap):
    SLY = feather.read_feather("Results/SLY.feather")
    SLY_fin = (SLY["Energy Carrier Net Total"] - SLY["Energy Carrier Net LOSS"]).unstack(level=0)
    E_K_fc_other = SLY_fin[["LYh", "LYg"]].sum(axis=1).unstack(level=["region cons"]).sum()
    E_K_fc_capital = SLY_fin[["LkYh", "LkYg"]].sum(axis=1).unstack(level=["region cons"]).sum() - E_K_fc_other

    pd.concat(
        [
            E_hhfin / pop_agg * 1000,
            E_K_fc_fin.sum() / pop_agg * 1000,
            E_K_fc_capital / pop_agg * 1000,
            E_K_fc_other / pop_agg * 1000,
            E_pbafin / pop_agg * 1000,
            E_cbafin / pop_agg * 1000,
            real_gdp_cap,
            fc_cap,
        ],
        axis=1,
        keys=[
            "E_hhfin (GJ/cap)",
            "E_K_fc_fin (GJ/cap)",
            "E_K_fc_capital_fin (GJ/cap)",
            "E_K_fc_other_fin (GJ/cap)",
            "E_pbafin (GJ/cap)",
            "E_cbafin (GJ/cap)",
            "GDP per capita (2017US$ppp)",
            "final consumption per capita (2017US$ppp)",
        ],
    ).to_excel("Figures/figure2.xlsx")


def data_figure3(real_gdp_cap, E_hhfin, pop_agg, E_cbafin):

    x = np.log(real_gdp_cap)
    y = np.log(E_hhfin / pop_agg * 1000)
    pd.concat([x, y], axis=1, keys=["log(GDP per capita (2017US\$ ppp))", "log($E_{hh}$(GJ/cap))"],).sort_values(
        by="log(GDP per capita (2017US\$ ppp))"
    ).to_excel("Figures/figure3.xlsx", sheet_name="Fig. 3a")
    regression = scipy.stats.linregress(x, y)
    with pd.ExcelWriter("Figures/figure3.xlsx", mode="a") as writer:
        pd.DataFrame(
            list(regression),
            index=["slope", "intercept", "rvalue", "pvalue", "stderr"],
            columns=["regression fig3a"],
        ).to_excel(writer, sheet_name="Fig. 3a regression")
    y = E_hhfin / (E_cbafin + E_hhfin) * 100
    regression = scipy.stats.linregress(x, y)
    with pd.ExcelWriter("Figures/figure3.xlsx", mode="a") as writer:
        pd.concat([x, y], axis=1, keys=["log(GDP per capita (2017US\$ ppp))", "Share (\%)"],).sort_values(
            by="log(GDP per capita (2017US\$ ppp))"
        ).to_excel(writer, sheet_name="Fig. 3b")
        pd.DataFrame(
            list(regression),
            index=["slope", "intercept", "rvalue", "pvalue", "stderr"],
            columns=["regression fig3b"],
        ).to_excel(writer, sheet_name="Fig. 3b regression")


def data_figure4(real_gdp_cap, E_pbafin, E_hhfin, ICP_data_real, E_K_fc_fin, Y_real):
    x = real_gdp_cap / 1000
    y = (E_pbafin + E_hhfin) / ICP_data_real["1000000:GROSS DOMESTIC PRODUCT"] / 1000
    pd.concat([x, y], axis=1, keys=["GDP per capita (2017US\$ ppp)", "$I_{pba}$ (MJ/2017US\$ppp)"],).sort_values(
        by="GDP per capita (2017US\$ ppp)"
    ).to_excel("Figures/figure4.xlsx", sheet_name="Fig. 4a")
    regression = scipy.stats.linregress(x, y)
    with pd.ExcelWriter("Figures/figure4.xlsx", mode="a") as writer:
        pd.DataFrame(
            list(regression),
            index=["slope", "intercept", "rvalue", "pvalue", "stderr"],
            columns=["regression fig4a"],
        ).to_excel(writer, sheet_name="Fig. 4a regression")
        #
        pd.DataFrame(
            np.log(y / ((E_pbafin + E_hhfin).sum() / ICP_data_real["1000000:GROSS DOMESTIC PRODUCT"].sum() / 1000)),
            columns=["$I_{pba}$ log difference to world mean"],
        ).to_excel(writer, sheet_name="Fig. 4d")

    y = E_K_fc_fin.sum() / Y_real.T.drop(sect_cap, axis=1).sum(axis=1)
    regression = scipy.stats.linregress(x, y)
    with pd.ExcelWriter("Figures/figure4.xlsx", mode="a") as writer:
        pd.concat([x, y], axis=1, keys=["GDP per capita (2017US\$ ppp)", "$I^K_{fc}$ (MJ/2017US\$ppp)"],).sort_values(
            by="GDP per capita (2017US\$ ppp)"
        ).to_excel(writer, sheet_name="Fig. 4c")
        pd.DataFrame(
            list(regression),
            index=["slope", "intercept", "rvalue", "pvalue", "stderr"],
            columns=["regression fig4c"],
        ).to_excel(writer, sheet_name="Fig. 4c regression")

        pd.DataFrame(  # world values
            [
                (E_pbafin + E_hhfin).sum() / ICP_data_real["1000000:GROSS DOMESTIC PRODUCT"].sum() / 1000,
                E_K_fc_fin.sum(axis=1).sum() / Y_real.T.drop(sect_cap, axis=1).sum(axis=1).sum(),
            ],
            index=["$I_{pba}$", "$I^K_{fc}$"],
            columns=["world values"],
        ).to_excel(writer, sheet_name="World values")

        pd.DataFrame(
            np.log(y / (E_K_fc_fin.sum().sum() / Y_real.T.drop(sect_cap, axis=1).sum(axis=1).sum())),
            columns=["$I^K_{fc}$ log difference to world mean"],
        ).to_excel(writer, sheet_name="Fig. 4e")

        # bar plot
        pd.concat(
            [
                E_hhfin / ICP_data_real["1000000:GROSS DOMESTIC PRODUCT"] / 1000,
                E_pbafin / ICP_data_real["1000000:GROSS DOMESTIC PRODUCT"] / 1000,
                E_K_fc_fin.sum() / Y_real.T.drop(sect_cap, axis=1).sum(axis=1),
            ],
            axis=1,
            keys=[
                "$E_{hh}/Y$ (MJ/2017US\$ppp)",
                "$E_{pba}/Y$ (MJ/2017US\$ppp)",
                "$E^K_{fc}/Y_{fc}$ (MJ/2017US\$ppp)",
            ],
        ).to_excel(writer, sheet_name="Fig. 4b")

        y = E_K_fc_fin.sum() / Y_real.T.drop(sect_cap, axis=1).sum(axis=1)
        regression = scipy.stats.linregress(np.log(x), np.log(y))
        pd.DataFrame(
            list(regression),
            index=["slope", "intercept", "rvalue", "pvalue", "stderr"],
            columns=["elasticity E_K_fc"],
        ).to_excel(writer, sheet_name="elasticity E_K_fc")

        y = (E_pbafin + E_hhfin) / ICP_data_real["1000000:GROSS DOMESTIC PRODUCT"] / 1000
        regression = scipy.stats.linregress(np.log(x), np.log(y))
        pd.DataFrame(
            list(regression),
            index=["slope", "intercept", "rvalue", "pvalue", "stderr"],
            columns=["elasticity E_pba"],
        ).to_excel(writer, sheet_name="elasticity E_pba")


def data_figure5(E_K_fc_fin, Y_real, real_gdp_cap):
    I_K_fc = E_K_fc_fin.T.div(Y_real.T.drop(sect_cap, axis=1))
    I_K_fc_world = E_K_fc_fin.sum(axis=1) / Y_real.T.drop(sect_cap, axis=1).sum()
    x = real_gdp_cap / 1000
    regression_results = pd.DataFrame([], index=["slope", "intercept", "rvalue", "pvalue", "stderr"])
    log_diff = pd.DataFrame([], index=I_K_fc.index, columns=I_K_fc.columns)
    for i in I_K_fc.columns:
        y = I_K_fc[i]
        regression = scipy.stats.linregress(x, y)
        regression_results[i] = list(regression)
        log_diff[i] = np.log(I_K_fc[i] / I_K_fc_world.loc[i])
    regression_results.loc["R2"] = regression_results.loc["rvalue"] * regression_results.loc["rvalue"]
    I_K_fc.to_excel("Figures/figure5.xlsx", sheet_name="I_K_fc")
    with pd.ExcelWriter("Figures/figure5.xlsx", mode="a") as writer:
        I_K_fc_world.to_excel(writer, sheet_name="I_K_fc_world")
        regression_results.to_excel(writer, sheet_name="regression_results")
        log_diff.to_excel(writer, sheet_name="log difference to world mean")
        x.to_excel(writer, sheet_name="real gdp cap")


def data_figure6(Y_real, real_gdp_cap):
    sect_shares = Y_real.drop(sect_cap).div(Y_real.drop(sect_cap).sum(), axis=1).T
    x = (real_gdp_cap) / 1000
    regression_results = pd.DataFrame([], index=["slope", "intercept", "rvalue", "pvalue", "stderr"])

    for i in sect_shares.columns:
        y = sect_shares[i]
        regression = scipy.stats.linregress(x, y)
        regression_results[i] = list(regression)
    regression_results.loc["R2"] = regression_results.loc["rvalue"] * regression_results.loc["rvalue"]
    sect_shares["real gdp cap"] = x
    sect_shares.to_excel("Figures/figure6.xlsx", sheet_name="sector shares")
    with pd.ExcelWriter("Figures/figure6.xlsx", mode="a") as writer:
        regression_results.to_excel(writer, sheet_name="regression_results")


def data_figure7(real_gdp_cap, E_K_fc_fin, Y_real, E_hhfin, pop_agg):

    x = real_gdp_cap
    y = E_K_fc_fin.sum() / Y_real.T.drop(sect_cap, axis=1).sum(axis=1)
    regression_fd = scipy.stats.linregress(x, y)
    y = E_hhfin / pop_agg
    regression_hh = scipy.stats.linregress(np.log(x), np.log(y))
    y = ((E_hhfin + E_K_fc_fin.sum()) / pop_agg) * 1000000
    ycalc = (regression_fd[1] + regression_fd[0] * x) * Y_real.T.drop(sect_cap, axis=1).sum(
        axis=1
    ) * 1000000 / pop_agg + np.exp(regression_hh[1] + regression_hh[0] * np.log(x)) * 1000000
    pd.concat([x, y, ycalc], axis=1, keys=["real_gdp_cap", "GJ/capita", "GJ/capita calculated"],).sort_values(
        by="real_gdp_cap"
    ).to_excel("Figures/figure7.xlsx")


def data_figure8(real_gdp_cap, E_K_fc_fin, Y_real, E_hhfin, pop_agg):

    long_exio_regions = dict(
        {
            "AT": "Austria",
            "BE": "Belgium",
            "BG": "Bulgaria",
            "CY": "Cyprus",
            "CZ": "Czech Republic",
            "DE": "Germany",
            "DK": "Denmark",
            "EE": "Estonia",
            "ES": "Spain",
            "FI": "Finland",
            "FR": "France",
            "GR": "Greece",
            "HR": "Croatia",
            "HU": "Hungary",
            "IE": "Ireland",
            "IT": "Italy",
            "LT": "Lithuania",
            "LU": "Luxembourg",
            "LV": "Latvia",
            "MT": "Malta",
            "NL": "Netherlands",
            "PL": "Poland",
            "PT": "Portugal",
            "RO": "Romania",
            "SE": "Sweden",
            "SI": "Slovenia",
            "SK": "Slovakia",
            "GB": "United Kingdom",
            "US": "United States",
            "JP": "Japan",
            "CN": "China",
            "CA": "Canada",
            "KR": "South Korea",
            "BR": "Brazil",
            "IN": "India",
            "MX": "Mexico",
            "RU": "Russia",
            "AU": "Australia",
            "CH": "Switzerland",
            "TR": "Turkey",
            "TW": "Taiwan",
            "NO": "Norway",
            "ID": "Indonesia",
            "ZA": "South Africa",
            "WA": "RoW Asia and Pacific",
            "WL": "RoW America",
            "WE": "RoW Europe",
            "WF": "RoW Africa",
            "WM": "RoW Middle East",
        }
    )

    x = real_gdp_cap
    y = E_K_fc_fin.sum() / Y_real.T.drop(sect_cap, axis=1).sum(axis=1)  # MJ/fc
    regression_fc = scipy.stats.linregress(x, y)
    a = regression_fc[0]
    b = y - a * x

    y = E_hhfin * 1000000 / pop_agg  # MJ/cap
    regression_hh = scipy.stats.linregress(np.log(x), np.log(y))

    y = (Y_real.T.drop(sect_cap, axis=1).sum(axis=1) / pop_agg * 1000000) * (0.01 * b + 0.0201 * real_gdp_cap * a) + (
        E_hhfin * 1000000 / pop_agg
    ) * (pow(1.01, regression_hh[0]) - 1)

    data = pd.concat([x, y], keys=["gdp", "delta"], axis=1).sort_values(by="gdp")
    data.rename(index=long_exio_regions).to_excel("Figures/figure8.xlsx")


def data_table1(E_K_fc_fin, Y_real, real_gdp_cap):
    I_K_fc = E_K_fc_fin.T.div(Y_real.T.drop(sect_cap, axis=1))
    I_K_fc_world = E_K_fc_fin.sum(axis=1) / Y_real.T.drop(sect_cap, axis=1).sum()
    I_K_fc_world_total = E_K_fc_fin.sum().sum() / Y_real.T.drop(sect_cap, axis=1).sum().sum()

    sect_shares = Y_real.drop(sect_cap).div(Y_real.drop(sect_cap).sum(), axis=1)
    sect_shares_world = Y_real.drop(sect_cap).sum(axis=1).div(Y_real.drop(sect_cap).sum(axis=1).sum())

    # theta

    df_theta = pd.DataFrame(
        index=sect_shares.index,
        columns=[
            "World value",
            "$R^2$",
            "$\epsilon_{sect}^{\\theta}$",
            "$\eta_{sect}^{\\theta}$",
        ],
    )
    for sect in sect_shares.index:
        df_theta["World value"].loc[sect] = round(sect_shares_world.loc[sect], 2)
        x = real_gdp_cap / 1000
        y = sect_shares.loc[sect]
        regression = scipy.stats.linregress(x, y)
        df_theta["$R^2$"].loc[sect] = round(regression[2] ** 2, 2)
        # if regression[2] ** 2 > 0.2:
        x = np.log(real_gdp_cap / 1000)
        y = np.log(sect_shares.loc[sect])
        regression = scipy.stats.linregress(x, y)
        df_theta["$\epsilon_{sect}^{\\theta}$"].loc[sect] = round(regression[0], 2)
        df_theta["$\eta_{sect}^{\\theta}$"].loc[sect] = round(
            sect_shares_world.loc[sect]
            * regression[0]
            * (I_K_fc_world.loc[sect] - I_K_fc_world_total)
            / I_K_fc_world_total,
            2,
        )

    df_I = pd.DataFrame(
        index=I_K_fc_world.index,
        columns=["World value", "$R^2$", "$\epsilon_{sect}^{I}$", "$\eta_{sect}^{I}$"],
    )
    for sect in I_K_fc.columns:
        df_I["World value"].loc[sect] = round(I_K_fc_world.loc[sect], 1)
        x = real_gdp_cap / 1000
        y = I_K_fc[sect]
        regression = scipy.stats.linregress(x, y)
        df_I["$R^2$"].loc[sect] = round(regression[2] ** 2, 2)
        # if regression[2] ** 2 > 0.2:
        x = np.log(real_gdp_cap / 1000)
        y = np.log(I_K_fc[sect])
        regression = scipy.stats.linregress(x, y)
        df_I["$\epsilon_{sect}^{I}$"].loc[sect] = round(regression[0], 2)
        df_I["$\eta_{sect}^{I}$"].loc[sect] = round(
            sect_shares_world.loc[sect] * regression[0] * I_K_fc_world.loc[sect] / I_K_fc_world_total,
            2,
        )

    df = pd.concat([df_I, df_theta], keys=[r"$I_{sect}^{K}$", "$\\theta_{sect}$"], axis=1)
    df.to_excel("Figures/tab1.xlsx")


# .....sub sector calculations.........

# this function calculates, for each sub-sector of the 12 CPI sectors,
# the regression of a sub sector's share of expenditures in the CPI sector total as a function of GDP
# it saves an excel with the coefficients of regressions and the shares
def supplementary_data_theta(real_gdp_cap, index, Yh_dollars, Yg_dollars):
    conc = pd.read_excel("concordance.xlsx", sheet_name="final consumption", index_col=0)
    sector_dict = dict(
        {
            "1101000:FOOD AND NON-ALCOHOLIC BEVERAGES": "CPI: 01 - Food and non-Alcoholic beverages",
            "1102000:ALCOHOLIC BEVERAGES, TOBACCO AND NARCOTICS": "CPI: 02 - Alcoholic beverages, tobacco and narcotics",
            "1103000:CLOTHING AND FOOTWEAR": "CPI: 03 - Clothing and footwear",
            "9060000:ACTUAL HOUSING, WATER, ELECTRICITY, GAS AND OTHER FUELS": "CPI: 04 - Housing, water, electricity, gas and other fuels",
            "1105000:FURNISHINGS, HOUSEHOLD EQUIPMENT AND ROUTINE HOUSEHOLD MAINTENANCE": "CPI: 05 - Furnishings, household equipment and routine household maintenance",
            "9080000:ACTUAL HEALTH": "CPI: 06 - Health",
            "1107000:TRANSPORT": "CPI: 07 - Transport",
            "1108000:COMMUNICATION": "CPI: 08 - Communication",
            "9110000:ACTUAL RECREATION AND CULTURE": "CPI: 09 - Recreation and culture",
            "9120000:ACTUAL EDUCATION": "CPI: 10 - Education",
            "1111000:RESTAURANTS AND HOTELS": "CPI: 11 - Restaurants and hotels",
            "9140000:ACTUAL MISCELLANEOUS GOODS AND SERVICES": "CPI: 12 - Miscellaneous goods and services",
            "9270000:GENERAL GOVERNMENT FINAL CONSUMPTION EXPENDITURE": "Final Consumption",
            "1501100:MACHINERY AND EQUIPMENT": "150100:MACHINERY AND EQUIPMENT",
            "1501200:CONSTRUCTION": "150200:CONSTRUCTION",
            "1501300:OTHER PRODUCTS": "150300:OTHER PRODUCTS",
        }
    )

    x = real_gdp_cap / 1000
    results = pd.DataFrame()
    shares = pd.DataFrame()

    for sect in conc.columns:
        regression_results = pd.DataFrame([], index=["slope", "intercept", "rvalue", "pvalue", "stderr"])
        df = (
            (Yh_dollars + Yg_dollars)
            .groupby(level="sector")
            .sum()
            .mul(conc[sect], axis=0)
            .div(index.rename(sector_dict)[sect], axis=1)
            .dropna()
        )
        df = df.loc[(df != 0).any(1)]
        share = pd.DataFrame()
        share[sect] = df.sum(axis=1) / df.sum().sum()
        df = df.div(df.sum(), axis=1)
        for sector in df.index:
            y = df.loc[sector]
            x2 = (x * y / y).dropna()
            y = (y * y / y).dropna()
            regression = scipy.stats.linregress(x2, y)
            regression_results[sector] = list(regression)
        regression_results.loc["R2"] = regression_results.loc["rvalue"] * regression_results.loc["rvalue"]
        regression_results2 = pd.DataFrame()
        regression_results2[sect] = regression_results.stack()

        results = pd.concat([results, regression_results2.unstack()], axis=1)
        shares = pd.concat([shares, share], axis=0)

    results.loc["shares (%)"] = shares.unstack().dropna() * 100

    results.loc[["R2", "shares (%)"]].T.loc[results.loc["R2"] > 0.2].round(2).to_excel(
        "Figures/supplementary_data_theta.xlsx", sheet_name="R2>0.2"
    )
    with pd.ExcelWriter("Figures/supplementary_data_theta.xlsx", mode="a") as writer:
        results.to_excel(writer, sheet_name="all sectors")


# this function calculates, for each sub-sector of the 12 CPI sectors,
# the regression of a sub sector's senergy intensity as a function of GDP
# it saves an excel with the coefficients of regressions and the shares
def supplementary_data_I(real_gdp_cap, index, Yh_dollars, Yg_dollars, Yh, Yg):
    conc = pd.read_excel("concordance.xlsx", sheet_name="final consumption", index_col=0)
    sector_dict = dict(
        {
            "1101000:FOOD AND NON-ALCOHOLIC BEVERAGES": "CPI: 01 - Food and non-Alcoholic beverages",
            "1102000:ALCOHOLIC BEVERAGES, TOBACCO AND NARCOTICS": "CPI: 02 - Alcoholic beverages, tobacco and narcotics",
            "1103000:CLOTHING AND FOOTWEAR": "CPI: 03 - Clothing and footwear",
            "9060000:ACTUAL HOUSING, WATER, ELECTRICITY, GAS AND OTHER FUELS": "CPI: 04 - Housing, water, electricity, gas and other fuels",
            "1105000:FURNISHINGS, HOUSEHOLD EQUIPMENT AND ROUTINE HOUSEHOLD MAINTENANCE": "CPI: 05 - Furnishings, household equipment and routine household maintenance",
            "9080000:ACTUAL HEALTH": "CPI: 06 - Health",
            "1107000:TRANSPORT": "CPI: 07 - Transport",
            "1108000:COMMUNICATION": "CPI: 08 - Communication",
            "9110000:ACTUAL RECREATION AND CULTURE": "CPI: 09 - Recreation and culture",
            "9120000:ACTUAL EDUCATION": "CPI: 10 - Education",
            "1111000:RESTAURANTS AND HOTELS": "CPI: 11 - Restaurants and hotels",
            "9140000:ACTUAL MISCELLANEOUS GOODS AND SERVICES": "CPI: 12 - Miscellaneous goods and services",
            "9270000:GENERAL GOVERNMENT FINAL CONSUMPTION EXPENDITURE": "Final Consumption",
            "1501100:MACHINERY AND EQUIPMENT": "150100:MACHINERY AND EQUIPMENT",
            "1501200:CONSTRUCTION": "150200:CONSTRUCTION",
            "1501300:OTHER PRODUCTS": "150300:OTHER PRODUCTS",
        }
    )
    x = real_gdp_cap / 1000
    results = pd.DataFrame()
    shares = pd.DataFrame()

    Lk = feather.read_feather("Results/Lk.feather")
    S = pd.read_csv(
        "Data/EXIO3/IOT_2017_pxp/satellite/S.txt",
        delimiter="\t",
        header=[0, 1],
        index_col=[0],
    )

    for sect in conc.columns:
        regression_results = pd.DataFrame([], index=["slope", "intercept", "rvalue", "pvalue", "stderr"])
        df = (
            (Yh_dollars + Yg_dollars)
            .groupby(level="sector")
            .sum()
            .mul(conc[sect], axis=0)
            .div(index.rename(sector_dict)[sect], axis=1)
        )

        df2 = df.loc[(df != 0).any(1)]
        share = pd.DataFrame()
        share[sect] = df2.sum(axis=1) / df2.sum().sum()

        SL = Lk.mul(S.loc["Energy Carrier Net Total"] - S.loc["Energy Carrier Net LOSS"], axis=0).sum()
        MJ = SL.unstack().T.mul((Yh + Yg).groupby(level="sector").sum().mul(conc[sect], axis=0))
        I = MJ / df  # .dropna()
        I = I.loc[(I != 0).any(1)]

        for sector in I.index:
            y = I.loc[sector]
            x2 = (x * y / y).dropna()
            y = (y * y / y).dropna()
            if x2.size != 0:
                regression = scipy.stats.linregress(x2, y)
                regression_results[sector] = list(regression)
        regression_results.loc["R2"] = regression_results.loc["rvalue"] * regression_results.loc["rvalue"]
        regression_results2 = pd.DataFrame()
        regression_results2[sect] = regression_results.stack()

        results = pd.concat([results, regression_results2.unstack()], axis=1)
        shares = pd.concat([shares, share], axis=0)

    results.loc["shares (%)"] = shares.unstack().dropna() * 100

    results.loc[["R2", "shares (%)"]].T.loc[results.loc["R2"] > 0.2].round(2).to_excel(
        "Figures/supplementary_data_I.xlsx", sheet_name="R2>0.2"
    )
    with pd.ExcelWriter("Figures/supplementary_data_I.xlsx", mode="a") as writer:
        results.to_excel(writer, sheet_name="all sectors")
