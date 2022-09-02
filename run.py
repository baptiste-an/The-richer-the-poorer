from functions import *
import zipfile

# download(
#     "https://zenodo.org/record/3874309/files/Kbar_exio_v3_6_2014pxp.mat",
#     dest_folder="Data/Sodersten",
# )
# download(
#     "https://zenodo.org/record/3874309/files/Kbar_exio_v3_6_2015pxp.mat",
#     dest_folder="Data/Sodersten",
# )
# download(
#     "https://zenodo.org/record/5589597/files/IOT_2017_pxp.zip", dest_folder="Data/EXIO3"
# )
# with zipfile.ZipFile("Data/EXIO3/IOT_2017_pxp.zip", "r") as zip_ref:
#     zip_ref.extractall("Data/EXIO3")


# ........CREATE KBAR MATRIX FOR YEAR 2017................

# Kbar()

# # .........CALCULATIONS......................................

# Y_all()  # function to diaggregate GDP into all the formats needed

# Lk()  # function to calculate Lk from Z and Kbar

# LY()  # function to calculate output associated with L and Lk for each component of Y

# SLY()  # associates energy extensions to LY

# ..............ICP CALCULATION..............

(
    real_gdp_cap,
    ICP_data_real,
    pop_agg,
    Y_real,
    fc_cap,
    index,
    Yh_dollars,
    Yg_dollars,
    Yh,
    Yg,
) = ppp_calculations()
# real_gdp_cap is the GDP for 49 regions in US$ppp
# ICP_data_real is the raw ICP data aggregated into 49 regions
# pop_agg is the population aggregated into 49 regions
# Y_real is the final consumption of households and governemnts (12 sectors) plus GFCF (3 sectors) for 49 regions in US$2017ppp
# fc_cap is the final consumption per capita in US$2017ppp/cap


# .........ENERGY INTENSITIES.............


# calculate all the energy vectors used to calculate energy intensities

(
    E_K_fc_fin,
    E_K_fc_pri,
    E_hhfin,
    E_hhpri,
    E_pbafin,
    E_pbapri,
    E_cbafin,
    E_cbapri,
) = energy()


# ..................VALIDATION......................

validation()  # close to zero if the energy footprint is the same with and without capital endogenized


# ..........DATA FOR FIGURES...................

data_figure1()
data_figure2(E_hhfin, pop_agg, E_K_fc_fin, E_pbafin, E_cbafin, real_gdp_cap, fc_cap)
data_figure3(real_gdp_cap, E_hhfin, pop_agg, E_cbafin)
data_figure4(real_gdp_cap, E_pbafin, E_hhfin, ICP_data_real, E_K_fc_fin, Y_real)
data_figure5(E_K_fc_fin, Y_real, real_gdp_cap)
data_figure6(Y_real, real_gdp_cap)
data_figure7(real_gdp_cap, E_K_fc_fin, Y_real, E_hhfin, pop_agg)
data_figure8(real_gdp_cap, E_K_fc_fin, Y_real, E_hhfin, pop_agg)
data_table1(E_K_fc_fin, Y_real, real_gdp_cap)

# ............SUPPLEMENTARY DATA...................

supplementary_data_theta(real_gdp_cap, index, Yh_dollars, Yg_dollars)

supplementary_data_I(real_gdp_cap, index, Yh_dollars, Yg_dollars, Yh, Yg)
