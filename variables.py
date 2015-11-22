import numpy as np
import pandas as pd

import orca
from urbansim.utils import misc
from urbansim_defaults import variables

import datasources

#####################
# COSTAR VARIABLES
#####################

@orca.column('costar', 'node_id')
def node_id(parcels, costar):
    return misc.reindex(parcels.node_id, costar.parcel_id)

@orca.column('costar', 'mgra_id')
def mgra_id(parcels, costar):
    return misc.reindex(parcels.mgra_id, costar.parcel_id)
    
@orca.column('costar', 'luz_id')
def luz_id(parcels, costar):
    return misc.reindex(parcels.luz_id, costar.parcel_id)
    
@orca.column('costar', 'distance_to_coast')
def distance_to_coast(parcels, costar):
    return misc.reindex(parcels.distance_to_coast, costar.parcel_id)
    
@orca.column('costar', 'distance_to_freeway')
def distance_to_freeway(parcels, costar):
    return misc.reindex(parcels.distance_to_freeway, costar.parcel_id)
    
@orca.column('costar', 'distance_to_onramp')
def distance_to_onramp(parcels, costar):
    return misc.reindex(parcels.distance_to_onramp, costar.parcel_id)
    
@orca.column('costar', 'distance_to_park')
def distance_to_park(parcels, costar):
    return misc.reindex(parcels.distance_to_park, costar.parcel_id)
    
@orca.column('costar', 'distance_to_school')
def distance_to_school(parcels, costar):
    return misc.reindex(parcels.distance_to_school, costar.parcel_id)
    
@orca.column('costar', 'distance_to_transit')
def distance_to_transit(parcels, costar):
    return misc.reindex(parcels.distance_to_transit, costar.parcel_id)
    
################################
# ASSESSOR TRANSACTION VARIABLES
################################

@orca.column('assessor_transactions', 'node_id')
def node_id(parcels, assessor_transactions):
    return misc.reindex(parcels.node_id, assessor_transactions.parcel_id)

@orca.column('assessor_transactions', 'mgra_id')
def mgra_id(parcels, assessor_transactions):
    return misc.reindex(parcels.mgra_id, assessor_transactions.parcel_id)
    
@orca.column('assessor_transactions', 'luz_id')
def luz_id(parcels, assessor_transactions):
    return misc.reindex(parcels.luz_id, assessor_transactions.parcel_id)
    
@orca.column('assessor_transactions', 'distance_to_coast')
def distance_to_coast(parcels, assessor_transactions):
    return misc.reindex(parcels.distance_to_coast, assessor_transactions.parcel_id)
    
@orca.column('assessor_transactions', 'distance_to_freeway')
def distance_to_freeway(parcels, assessor_transactions):
    return misc.reindex(parcels.distance_to_freeway, assessor_transactions.parcel_id)
    
@orca.column('assessor_transactions', 'distance_to_onramp')
def distance_to_onramp(parcels, assessor_transactions):
    return misc.reindex(parcels.distance_to_onramp, assessor_transactions.parcel_id)
    
@orca.column('assessor_transactions', 'distance_to_park')
def distance_to_park(parcels, assessor_transactions):
    return misc.reindex(parcels.distance_to_park, assessor_transactions.parcel_id)
    
@orca.column('assessor_transactions', 'distance_to_school')
def distance_to_school(parcels, assessor_transactions):
    return misc.reindex(parcels.distance_to_school, assessor_transactions.parcel_id)
    
@orca.column('assessor_transactions', 'distance_to_transit')
def distance_to_transit(parcels, assessor_transactions):
    return misc.reindex(parcels.distance_to_transit, assessor_transactions.parcel_id)
    
@orca.column('assessor_transactions', 'sqft_per_unit')
def sqft_per_unit(assessor_transactions):
    sqft_per_unit = pd.Series(np.zeros(len(assessor_transactions))*1.0, index = assessor_transactions.index)
    sqft_per_unit[assessor_transactions.residential_units > 0] = assessor_transactions.residential_sqft[assessor_transactions.residential_units > 0] / assessor_transactions.residential_units[assessor_transactions.residential_units > 0]
    return sqft_per_unit
    
@orca.column('assessor_transactions', 'year_built_1940to1950')
def year_built_1940to1950(assessor_transactions):
    return (assessor_transactions.year_built >= 1940) & (assessor_transactions.year_built < 1950)
    
@orca.column('assessor_transactions', 'year_built_1950to1960')
def year_built_1950to1960(assessor_transactions):
    return (assessor_transactions.year_built >= 1950) & (assessor_transactions.year_built < 1960)
    
@orca.column('assessor_transactions', 'year_built_1960to1970')
def year_built_1960to1970(assessor_transactions):
    return (assessor_transactions.year_built >= 1960) & (assessor_transactions.year_built < 1970)
    
@orca.column('assessor_transactions', 'year_built_1970to1980')
def year_built_1970to1980(assessor_transactions):
    return (assessor_transactions.year_built >= 1970) & (assessor_transactions.year_built < 1980)
    
@orca.column('assessor_transactions', 'year_built_1980to1990')
def year_built_1980to1990(assessor_transactions):
    return (assessor_transactions.year_built >= 1980) & (assessor_transactions.year_built < 1990)
    
    
####NODES
    
@orca.column('nodes', 'nonres_occupancy_3000m')
def nonres_occupancy_3000m(nodes):
    return nodes.jobs_3000m / (nodes.job_spaces_3000m + 1.0)
    
@orca.column('nodes', 'res_occupancy_3000m')
def res_occupancy_3000m(nodes):
    return nodes.households_3000m / (nodes.residential_units_3000m + 1.0)
    
    
#####################
# BUILDINGS VARIABLES
#####################

@orca.column('buildings', 'is_office', cache=True, cache_scope='iteration')
def is_office(buildings):
    return (buildings.development_type_id == 4).astype('int')
    
@orca.column('buildings', 'is_retail', cache=True, cache_scope='iteration')
def is_retail(buildings):
    return (buildings.development_type_id == 5).astype('int')

@orca.column('buildings', 'job_spaces', cache=True, cache_scope='iteration')
def job_spaces():
    store = orca.get_injectable('store')
    b = orca.get_table('buildings').to_frame(['luz_id', 'development_type_id','non_residential_sqft', 'year_built'])
    bsqft_job = store['building_sqft_per_job']
    merged = pd.merge(b.reset_index(), bsqft_job, left_on = ['luz_id', 'development_type_id'], right_on = ['luz_id', 'development_type_id'])
    merged = merged.set_index('building_id')
    merged.sqft_per_emp[merged.sqft_per_emp < 40] = 40
    merged['job_spaces'] = np.ceil(merged.non_residential_sqft / merged.sqft_per_emp)
    job_spaces = pd.Series(merged.job_spaces, index = b.index)
    b['job_spaces'] = job_spaces
    b.job_spaces[(b.luz_id <17)&(b.year_built<2013)] = np.ceil(b.job_spaces[(b.luz_id <17)&(b.year_built<2013)]/10.0)
    b.job_spaces[(b.job_spaces > 2000)&(b.year_built<2013)] = 2000
    b.job_spaces[b.job_spaces.isnull()] = np.ceil(b.non_residential_sqft/200.0)
    b.job_spaces[b.year_built < 2013] = np.ceil(b.job_spaces[b.year_built < 2013]/3.25)
    return b.job_spaces

@orca.column('buildings', 'luz_id')
def luz_id(buildings, parcels):
    return misc.reindex(parcels.luz_id, buildings.parcel_id)
    
@orca.column('buildings', 'luz_id_buildings')
def luz_id_buildings(buildings, parcels):
    return misc.reindex(parcels.luz_id, buildings.parcel_id)
    
@orca.column('buildings', 'node_id', cache=True, cache_scope='iteration')
def node_id(parcels, buildings):
    return misc.reindex(parcels.node_id, buildings.parcel_id)

@orca.column('buildings', 'mgra_id', cache=True, cache_scope='iteration')
def mgra_id(parcels, buildings):
    return misc.reindex(parcels.mgra_id, buildings.parcel_id)
    
@orca.column('buildings', 'zone_id', cache=True, cache_scope='iteration')
def zone_id(buildings):
    return np.zeros(len(buildings))
    
@orca.column('buildings', 'msa_id', cache=True, cache_scope='iteration')
def msa_id(buildings, parcels):
    return misc.reindex(parcels.msa_id, buildings.parcel_id)
    
@orca.column('buildings', 'parcel_size', cache=True, cache_scope='iteration')
def parcel_size(buildings):
    return np.zeros(len(buildings))
    
@orca.column('buildings', 'building_sqft', cache=True, cache_scope='iteration')
def building_sqft(buildings):
    return buildings.residential_sqft + buildings.non_residential_sqft
    
@orca.column('buildings', 'sqft_per_unit', cache=True, cache_scope='iteration')
def sqft_per_unit(buildings):
    sqft_per_unit = pd.Series(np.zeros(len(buildings))*1.0, index = buildings.index)
    sqft_per_unit[buildings.residential_units > 0] = buildings.residential_sqft[buildings.residential_units > 0] / buildings.residential_units[buildings.residential_units > 0]
    return sqft_per_unit
    
@orca.column('buildings', 'vacant_residential_units')
def vacant_residential_units(buildings, households):
    return buildings.residential_units.sub(
        households.building_id.value_counts(), fill_value=0)
    
@orca.column('buildings', 'building_type_id', cache=True, cache_scope='iteration')
def building_type_id(buildings):
    return buildings.development_type_id
    
@orca.column('buildings', 'distance_to_coast', cache=True, cache_scope='iteration')
def distance_to_coast(parcels, buildings):
    return misc.reindex(parcels.distance_to_coast, buildings.parcel_id)
    
@orca.column('buildings', 'distance_to_freeway', cache=True, cache_scope='iteration')
def distance_to_freeway(parcels, buildings):
    return misc.reindex(parcels.distance_to_freeway, buildings.parcel_id)
    
@orca.column('buildings', 'distance_to_onramp', cache=True, cache_scope='iteration')
def distance_to_onramp(parcels, buildings):
    return misc.reindex(parcels.distance_to_onramp, buildings.parcel_id)
    
@orca.column('buildings', 'distance_to_park', cache=True, cache_scope='iteration')
def distance_to_park(parcels, buildings):
    return misc.reindex(parcels.distance_to_park, buildings.parcel_id)
    
@orca.column('buildings', 'distance_to_school', cache=True, cache_scope='iteration')
def distance_to_school(parcels, buildings):
    return misc.reindex(parcels.distance_to_school, buildings.parcel_id)
    
@orca.column('buildings', 'distance_to_transit', cache=True, cache_scope='iteration')
def distance_to_transit(parcels, buildings):
    return misc.reindex(parcels.distance_to_transit, buildings.parcel_id)
    
@orca.column('buildings', 'year_built_1940to1950', cache=True, cache_scope='iteration')
def year_built_1940to1950(buildings):
    return (buildings.year_built >= 1940) & (buildings.year_built < 1950)
    
@orca.column('buildings', 'year_built_1950to1960', cache=True, cache_scope='iteration')
def year_built_1950to1960(buildings):
    return (buildings.year_built >= 1950) & (buildings.year_built < 1960)
    
@orca.column('buildings', 'year_built_1960to1970', cache=True, cache_scope='iteration')
def year_built_1960to1970(buildings):
    return (buildings.year_built >= 1960) & (buildings.year_built < 1970)
    
@orca.column('buildings', 'year_built_1970to1980', cache=True, cache_scope='iteration')
def year_built_1970to1980(buildings):
    return (buildings.year_built >= 1970) & (buildings.year_built < 1980)
    
@orca.column('buildings', 'year_built_1980to1990', cache=True, cache_scope='iteration')
def year_built_1980to1990(buildings):
    return (buildings.year_built >= 1980) & (buildings.year_built < 1990)
    
#####################
# JOB VARIABLES
#####################

@orca.column('jobs', 'luz_id')
def luz_id(jobs, buildings):
    return misc.reindex(buildings.luz_id, jobs.building_id)
    
#####################
# HOUSEHOLD VARIABLES
#####################

@orca.column('households', 'luz_id')
def luz_id(households, buildings):
    return misc.reindex(buildings.luz_id, households.building_id)
    
@orca.column('households', 'mgra_id', cache=True, cache_scope='iteration')
def mgra_id(households, buildings):
    return misc.reindex(buildings.mgra_id, households.building_id)
    
@orca.column('households', 'luz_id_households', cache=True, cache_scope='iteration')
def luz_id_households(households, buildings):
    return misc.reindex(buildings.luz_id, households.building_id)
    
@orca.column('households', 'activity_id', cache=True, cache_scope='iteration')
def activity_id(households):
    idx_38 = (households.income < 25000) & (households.persons < 3)
    idx_39 = (households.income < 25000) & (households.persons >= 3)
    idx_40 = (households.income >= 25000) & (households.income < 150000) & (households.persons < 3)
    idx_41 = (households.income >= 25000) & (households.income < 150000) & (households.persons >= 3)
    idx_42 = (households.income >= 150000) & (households.persons < 3)
    idx_43 = (households.income >= 150000) & (households.persons >= 3)
    return 38*idx_38 + 39*idx_39 + 40*idx_40 + 41*idx_41 + 42*idx_42 + 43*idx_43
    
@orca.column('households', 'income_halves', cache=True, cache_scope='iteration')
def income_halves(households):
    s = pd.Series(pd.qcut(households.income, 2, labels=False),
                  index=households.index)
    s = s.add(1)
    return s
    
#####################
# PARCEL VARIABLES
#####################
        
@orca.column('parcels', 'parcel_acres', cache=True)
def parcel_acres(parcels):
    return parcels.acres
        
@orca.column('parcels', 'parcel_size', cache=True)
def parcel_size(parcels):
    return parcels.parcel_acres * 43560
    
@orca.column('parcels', 'proportion_developable', cache=True)
def proportion_developable(parcels):
    parcels = parcels.to_frame(columns = ['development_type_id', 'proportion_undevelopable'])
    parcels.proportion_undevelopable[parcels.development_type_id == 24] = 1.0 # Set right-of-way parcels as undevelopable
    return 1.0 - parcels.proportion_undevelopable
    
@orca.injectable('parcel_is_allowed_func', autocall=False)
def parcel_is_allowed(form):
    parcels = orca.get_table('parcels')
    zoning_allowed_uses = orca.get_table('zoning_allowed_uses').to_frame()
    
    if form == 'sf_detached':
        allowed = zoning_allowed_uses[19]
    elif form == 'sf_attached':
        allowed = zoning_allowed_uses[20]
    elif form == 'mf_residential':
        allowed = zoning_allowed_uses[21]
    elif form == 'light_industrial':
        allowed = zoning_allowed_uses[2]
    elif form == 'heavy_industrial':
        allowed = zoning_allowed_uses[3]
    elif form == 'office':
        allowed = zoning_allowed_uses[4]
    elif form == 'retail':
        allowed = zoning_allowed_uses[5]
    else:
        df = pd.DataFrame(index=parcels.index)
        df['allowed'] = True
        allowed = df.allowed
        
    return allowed
    
@orca.injectable('parcel_sales_price_sqft_func', autocall=False)
def parcel_sales_price_sqft(use):
    s = parcel_average_price(use)
    if use == "residential": s *= 1.2
    return s
    
@orca.injectable('parcel_average_price', autocall=False)
def parcel_average_price(use):
    return misc.reindex(orca.get_table('nodes')[use],
                        orca.get_table('parcels').node_id)
                        
@orca.column('parcels', 'max_dua', cache=True)
def max_dua(parcels, zoning):
    sr = misc.reindex(zoning.max_dua, parcels.zoning_id)
    sr = sr*parcels.proportion_developable
    df = pd.DataFrame({'max_dua':sr.values}, index = sr.index.values)
    df['index'] = df.index.values
    df = df.drop_duplicates()
    del df['index']
    df.index.name = 'parcel_id'
    return df.max_dua
    
@orca.column('parcels', 'max_far', cache=True)
def max_far(parcels, zoning):
    sr = misc.reindex(zoning.max_far, parcels.zoning_id)
    sr = sr*parcels.proportion_developable
    df = pd.DataFrame({'max_far':sr.values}, index = sr.index.values)
    df['index'] = df.index.values
    df = df.drop_duplicates()
    del df['index']
    df.index.name = 'parcel_id'
    return df.max_far
    
##Placeholder-  building height currently unconstrained (very high limit-  1000 ft.)
@orca.column('parcels', 'max_height', cache=True)
def max_height(parcels):
    return pd.Series(np.ones(len(parcels))*1000.0, index = parcels.index)
    
    
@orca.column('parcels', 'total_sqft')
def total_sqft(parcels, buildings):
    return buildings.building_sqft.groupby(buildings.parcel_id).sum().\
        reindex(parcels.index).fillna(0)
    

@orca.column('parcels', 'building_purchase_price_sqft', cache=True, cache_scope='iteration')
def building_purchase_price_sqft():
    return parcel_average_price("residential") * .81


@orca.column('parcels', 'building_purchase_price', cache=True, cache_scope='iteration')
def building_purchase_price(parcels):
    return (parcels.total_sqft * parcels.building_purchase_price_sqft).\
        reindex(parcels.index).fillna(0)

@orca.column('parcels', 'land_cost', cache=True, cache_scope='iteration')
def land_cost(parcels):
    return parcels.building_purchase_price + parcels.parcel_acres * 43560 * 12.21
    
@orca.column('parcels', 'total_sfd_du')
def total_sfd_du(parcels, buildings):
    buildings = buildings.to_frame(buildings.local_columns)
    return buildings[buildings.development_type_id == 19].residential_units.groupby(buildings.parcel_id).sum().\
        reindex(parcels.index).fillna(0)
        
@orca.column('parcels', 'total_sfa_du')
def total_sfa_du(parcels, buildings):
    buildings = buildings.to_frame(buildings.local_columns)
    return buildings[buildings.development_type_id == 20].residential_units.groupby(buildings.parcel_id).sum().\
        reindex(parcels.index).fillna(0)
        
@orca.column('parcels', 'total_mfr_du')
def total_mfr_du(parcels, buildings):
    buildings = buildings.to_frame(buildings.local_columns)
    return buildings[buildings.development_type_id == 21].residential_units.groupby(buildings.parcel_id).sum().\
        reindex(parcels.index).fillna(0)
        
@orca.column('parcels', 'newest_building')
def newest_building(parcels, buildings):
    return buildings.year_built.groupby(buildings.parcel_id).max().\
        reindex(parcels.index).fillna(0)