import os
import cStringIO

import numpy as np
import pandas as pd
import pandas.io.sql as sql

import orca
import pandana as pdna
from urbansim.utils import misc
from urbansim_defaults import utils
from urbansim_defaults import models
from urbansim.models import transition

import variables
import datasources


@orca.step('build_networks')
def build_networks(parcels):
    st = pd.HDFStore(os.path.join(misc.data_dir(), "osm_sandag.h5"), "r")
    nodes, edges = st.nodes, st.edges
    net = pdna.Network(nodes["x"], nodes["y"], edges["from"], edges["to"],
                       edges[["weight"]])
    net.precompute(3000)
    orca.add_injectable("net", net)
    
    p = parcels.to_frame(parcels.local_columns)
    p['node_id'] = net.get_node_ids(p['x'], p['y'])
    orca.add_table("parcels", p)

@orca.step('households_transition')
def households_transition(households, annual_household_control_totals, year):
    ct = annual_household_control_totals.to_frame()
    tran = transition.TabularTotalsTransition(ct, 'total_number_of_households')
    model = transition.TransitionModel(tran)
    hh = households.to_frame(households.local_columns + ['activity_id'])
    new, added_hh_idx, empty_dict = \
        model.transition(hh, year,)
    new.loc[added_hh_idx, "building_id"] = -1
    orca.add_table("households", new)
    
@orca.step('households_transition_basic')
def households_transition_basic(households):
    return utils.simple_transition(households, .01, "building_id")
    
@orca.step('jobs_transition')
def jobs_transition(jobs):
    return utils.simple_transition(jobs, .01, "building_id")
    
@orca.step('nrh_estimate2')
def nrh_estimate2(costar, aggregations):
    return utils.hedonic_estimate("nrh2.yaml", costar, aggregations)


@orca.step('nrh_simulate2')
def nrh_simulate2(buildings, aggregations):
    return utils.hedonic_simulate("nrh2.yaml", buildings, aggregations,
                                  "nonres_rent_per_sqft")
                                  
@orca.step('rsh_estimate')
def rsh_estimate(assessor_transactions, aggregations):
    return utils.hedonic_estimate("rsh.yaml", assessor_transactions, aggregations)
    
@orca.step('rsh_simulate')
def rsh_simulate(buildings, aggregations):
    return utils.hedonic_simulate("rsh.yaml", buildings, aggregations,
                                  "res_price_per_sqft")

@orca.step('nrh_simulate')
def nrh_simulate(buildings, aggregations):
    return utils.hedonic_simulate("nrh.yaml", buildings, aggregations,
                                  "nonres_rent_per_sqft")
                                  
def get_year():
    year = orca.get_injectable('year')
    if year is None:
        year = 2012
    return year
    
@orca.step('scheduled_development_events')
def scheduled_development_events(buildings, scheduled_development_events):
    year = get_year()
    sched_dev = scheduled_development_events.to_frame()
    sched_dev = sched_dev[sched_dev.year_built==year]
    sched_dev['residential_sqft'] = sched_dev.sqft_per_unit*sched_dev.residential_units
    sched_dev['job_spaces'] = sched_dev.non_residential_sqft/400
    if len(sched_dev) > 0:
        max_bid = buildings.index.values.max()
        idx = np.arange(max_bid + 1,max_bid+len(sched_dev)+1)
        sched_dev['building_id'] = idx
        sched_dev = sched_dev.set_index('building_id')
        from urbansim.developer.developer import Developer
        merge = Developer(pd.DataFrame({})).merge
        b = buildings.to_frame(buildings.local_columns)
        all_buildings = merge(b,sched_dev[b.columns])
        orca.add_table("buildings", all_buildings)
        
@orca.step('model_integration_indicators')
def model_integration_indicators():
    year = get_year()
    
    #Households by MGRA
    print 'Exporting indicators: households by MGRA'
    hh = orca.get_table('households')
    hh = hh.to_frame(hh.local_columns + ['mgra_id', 'activity_id'])
    mgra_indicators = hh.groupby(['mgra_id', 'activity_id']).size().reset_index()
    mgra_indicators.columns = ['mgra_id', 'activity_id', 'number_of_households']
    mgra_indicators.to_csv('./data/mgra_hh_%s.csv'%year, index = False)
    
    #Space by LUZ
    print 'Exporting indicators: space by LUZ'
    b = orca.get_table('buildings')
    b = b.to_frame(b.local_columns + ['luz_id'])
    luz_res_indicators = b[b.residential_units > 0].groupby(['luz_id', 'development_type_id']).residential_units.sum().reset_index()
    luz_res_indicators.columns = ['luz_id', 'development_type_id', 'residential_units']
    luz_res_indicators.to_csv('./data/luz_du_%s.csv'%year, index = False)
    
    luz_nonres_indicators = b[b.non_residential_sqft > 0].groupby(['luz_id', 'development_type_id']).non_residential_sqft.sum().reset_index()
    luz_nonres_indicators.columns = ['luz_id', 'development_type_id', 'non_residential_sqft']
    luz_nonres_indicators.to_csv('./data/luz_nrsf_%s.csv'%year, index = False)
    
@orca.step('buildings_to_uc')
def buildings_to_uc(buildings):
    year = get_year()
    
    # Export newly predicted buildings (from proforma or Sitespec) to Urban Canvas
    b = buildings.to_frame(buildings.local_columns)
    new_buildings =  b[(b.note=='simulated') | (b.note.str.startswith('Sitespec'))]
    new_buildings = new_buildings[new_buildings.year_built == year]
    new_buildings = new_buildings.reset_index()
    new_buildings = new_buildings.rename(columns = {'development_type_id':'building_type_id'})
    new_buildings['building_sqft'] = new_buildings.residential_sqft + new_buildings.non_residential_sqft
    new_buildings['sqft_per_unit'] =  new_buildings.residential_sqft/new_buildings.residential_units
    del new_buildings['res_price_per_sqft']
    del new_buildings['nonres_rent_per_sqft']
    new_buildings.parcel_id = new_buildings.parcel_id.astype('int32')
    new_buildings.residential_units = new_buildings.residential_units.astype('int32')
    new_buildings.non_residential_sqft = new_buildings.non_residential_sqft.astype('int32')
    new_buildings.stories = new_buildings.stories.astype('int32')
    new_buildings.residential_sqft = new_buildings.residential_sqft.astype('int32')
    new_buildings.building_sqft = new_buildings.building_sqft.fillna(0).astype('int32')
    new_buildings.sqft_per_unit = new_buildings.sqft_per_unit.fillna(0).astype('int32')
    
    # Urban Canvas database connection
    conn_string = ""
    
    if 'uc_conn' not in orca._INJECTABLES.keys():
        conn=psycopg2.connect(conn_string)
        cur = conn.cursor()
        
        orca.add_injectable('uc_conn', conn)
        orca.add_injectable('uc_cur', cur)
        
    else:
        conn = orca.get_injectable('uc_conn')
        cur = orca.get_injectable('uc_cur')
        
    def exec_sql_uc(query):
        try:
            cur.execute(query)
            conn.commit()
        except:
            conn=psycopg2.connect(conn_string)
            cur = conn.cursor()
            orca.add_injectable('uc_conn', conn)
            orca.add_injectable('uc_cur', cur)
            cur.execute(query)
            conn.commit()
            
    def get_val_from_uc_db(query):
        try:
            result = sql.read_frame(query, conn)
            return result.values[0][0]
        except:
            conn=psycopg2.connect(conn_string)
            cur = conn.cursor()
            orca.add_injectable('uc_conn', conn)
            orca.add_injectable('uc_cur', cur)
            result = sql.read_frame(query, conn)
            return result.values[0][0]
        
    max_bid = get_val_from_uc_db("select max(building_id) FROM building where building_id<100000000;")
    new_buildings.building_id = np.arange(max_bid+1, max_bid+1+len(new_buildings))

    if 'projects_num' not in orca._INJECTABLES.keys(): 
        exec_sql_uc("INSERT INTO scenario(id, name, type) select nextval('scenario_id_seq'), 'Run #' || cast(currval('scenario_id_seq') as character varying), 1;")
        nextval = get_val_from_uc_db("SELECT MAX(ID) FROM SCENARIO WHERE ID < 100000;")
        orca.add_injectable('projects_num', nextval)
        
        exec_sql_uc("INSERT INTO scenario_project(scenario, project) VALUES(%s, 1);" % nextval)
        exec_sql_uc("INSERT INTO scenario_project(scenario, project) VALUES(%s, %s);" % (nextval,nextval))
        
    else:
        nextval = orca.get_injectable('projects_num')

    nextval = '{' + str(nextval) + '}'
    new_buildings['projects'] = nextval

    valid_from = '{' + str(year) + '-1-1}'
    new_buildings['valid_from'] = valid_from
    print 'Exporting %s buildings to Urban Canvas database for project %s and year %s.' % (len(new_buildings),nextval,year)
    output = cStringIO.StringIO()
    new_buildings.to_csv(output, sep='\t', header=False, index=False)
    output.seek(0)
    cur.copy_from(output, 'building', columns =tuple(new_buildings.columns.values.tolist()))
    conn.commit()
    
@orca.injectable("add_extra_columns_func", autocall=False)
def add_extra_columns(df):
    for col in ['improvement_value', 'res_price_per_sqft', 'nonres_rent_per_sqft']:
        df[col] = 0.0
    df["note"] = 'simulated'
    df["year_built"] = get_year()
    df['dua'] = df.residential_units / (df.parcel_size / 43560.0)
    df['development_type_id'] = 0
    df.development_type_id[(df.form == 'residential') & (df.dua < 12)] = 19
    df.development_type_id[(df.form == 'residential') & (df.dua >= 12) & (df.dua < 25)] = 20
    df.development_type_id[(df.form == 'residential') & (df.dua >= 26)] = 21
    df.development_type_id[(df.form == 'retail')] = 5
    df.development_type_id[(df.form == 'industrial')] = 2
    df.development_type_id[(df.form == 'office')] = 4
    return df
    
@orca.step('luz_indicators')
def luz_indicators():
    bsim = orca.get_table('buildings').to_frame(columns = ['luz_id', 'note', 'res_price_per_sqft', 'nonres_rent_per_sqft', 'residential_units', 'non_residential_sqft'])
    luz_res_price = bsim[bsim.residential_units > 0].groupby('luz_id').res_price_per_sqft.mean()
    luz_nonres_price = bsim[bsim.non_residential_sqft > 0].groupby('luz_id').nonres_rent_per_sqft.mean()
    bsim = bsim[(bsim.note == 'simulated') | bsim.note.str.startswith('Sitespec')]
    luz_simdu = bsim.groupby('luz_id').residential_units.sum()
    luz_simnr = bsim.groupby('luz_id').non_residential_sqft.sum()
    luz_df = pd.DataFrame({'du':luz_simdu, 'nrsf':luz_simnr, 'res_price':luz_res_price, 'nonres_price':luz_nonres_price})
    luz_df = luz_df[luz_df.index.values != 0].fillna(0)
    luz_df.index = luz_df.index.values.astype('int')
    luz_df.index.name = 'luz_id'

    luz_base_indicators = orca.get_table('luz_base_indicators').to_frame()
    hh_sim = orca.get_table('households').to_frame(columns = ['luz_id'])
    emp_sim = orca.get_table('jobs').to_frame(columns = ['luz_id'])
    luz_df['hh_diff'] = hh_sim.groupby('luz_id').size().fillna(0) - luz_base_indicators.hh_base.fillna(0)
    luz_df['emp_diff'] = emp_sim.groupby('luz_id').size().fillna(0) - luz_base_indicators.emp_base.fillna(0)
    luz_df = luz_df.fillna(0)
    print luz_df.sum()
    luz_df.to_csv('./data/luz_sim_indicators.csv')
    

@orca.step('msa_indicators')
def msa_indicators():
    # Summarize results at MSA level
    b = orca.get_table('buildings').to_frame(columns = ['msa_id', 'mgra_id', 'residential_units', 'non_residential_sqft', 'note'])
    new_du_by_msa = b[b.note  == 'simulated'].groupby('msa_id').residential_units.sum()
    new_nrsf_by_msa = b[b.note  == 'simulated'].groupby('msa_id').non_residential_sqft.sum()
    proportion_du_by_msa = new_du_by_msa / new_du_by_msa.sum()
    proportion_nrsf_by_msa = new_nrsf_by_msa / new_nrsf_by_msa.sum()
    print proportion_du_by_msa
    print proportion_nrsf_by_msa 

# this if the function for mapping a specific building that we build to a
# specific building type
@orca.injectable("form_to_btype_func", autocall=False)
def form_to_btype_func(building):
    settings = orca.get_injectable('settings')
    form = building.form
    dua = building.residential_units / (building.parcel_size / 43560.0)
    # precise mapping of form to building type for residential
    if form is None or form == "residential":
        if dua < 12:
            return 19
        elif dua < 25:
            return 20
        return 21
    return settings["form_to_btype"][form][0]