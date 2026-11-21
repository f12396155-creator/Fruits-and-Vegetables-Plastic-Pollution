#!/usr/bin/env python3
"""Recalculate the morphology-corrected field microplastic mass reference used for Fig. 3b.

Input: Dataset_S3_Field_Microplastic_Mass_Estimation.xlsx
The workbook must contain Particle_Bin_Conversions, Condition_Level_Estimates,
Study_Level_Reference, Field_Reference_Statistics, and Geometry_Parameters.

The script recalculates per-bin morphology contributions, aggregates conditions,
calculates one median per study, and reports study-level P10/P50/P90.
"""
from pathlib import Path
import argparse, ast, json, math
import numpy as np
import pandas as pd

FILM_THICKNESS_MM = 0.01006
FRAGMENT_THICKNESS_MM = 0.02986
FIBER_DIAMETER_MM = 0.01990
DENSITY_G_CM3 = 1.0

def particle_mass_contributions(row):
    n = float(row['Particle_Abundance_items_per_kg'])
    d = float(row['Representative_Size_mm'])
    shares = ast.literal_eval(row['Morphology_Proportions'])
    film_share = float(shares.get('film', 0.0))
    fragment_share = float(shares.get('fragment', 0.0))
    fiber_share = float(shares.get('fiber', 0.0))
    spherical_share = sum(float(shares.get(k, 0.0)) for k in ('bead', 'sphere', 'pellet', 'foam'))
    film = n * film_share * (d ** 2 / 2.0) * FILM_THICKNESS_MM * DENSITY_G_CM3
    fragment = n * fragment_share * (d ** 2 / 2.0) * FRAGMENT_THICKNESS_MM * DENSITY_G_CM3
    fiber = n * fiber_share * math.pi * (FIBER_DIAMETER_MM / 2.0) ** 2 * d * DENSITY_G_CM3
    spherical = n * spherical_share * math.pi * d ** 3 / 6.0 * DENSITY_G_CM3
    return film, fiber, fragment, spherical

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='Data/Dataset_S3_Field_Microplastic_Mass_Estimation.xlsx')
    ap.add_argument('--outdir', default='Fig3b_recalculated')
    args = ap.parse_args()
    src = Path(args.input)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    bins = pd.read_excel(src, sheet_name='Particle_Bin_Conversions')
    cond_ref = pd.read_excel(src, sheet_name='Condition_Level_Estimates')
    study_ref = pd.read_excel(src, sheet_name='Study_Level_Reference')
    stats_ref = pd.read_excel(src, sheet_name='Field_Reference_Statistics')
    assert len(bins) == 156 and bins['Study_ID'].nunique() == 13
    calc = bins.copy()
    vals = calc.apply(particle_mass_contributions, axis=1, result_type='expand')
    vals.columns = ['Film_Recalculated', 'Fiber_Recalculated', 'Fragment_Recalculated', 'Spherical_Recalculated']
    calc = pd.concat([calc, vals], axis=1)
    calc['Recalculated_Total_mg_per_kg'] = calc[['Film_Recalculated','Fiber_Recalculated','Fragment_Recalculated','Spherical_Recalculated']].sum(axis=1)
    # Verify the recomputed bin masses against the submitted source-data values.
    for new, old in [('Film_Recalculated','Film_Mass_Contribution_mg_per_kg'),('Fiber_Recalculated','Fiber_Mass_Contribution_mg_per_kg'),('Fragment_Recalculated','Fragment_Mass_Contribution_mg_per_kg'),('Spherical_Recalculated','Spherical_Particle_Mass_Contribution_mg_per_kg'),('Recalculated_Total_mg_per_kg','Morphology_Corrected_Total_mg_per_kg')]:
        if not np.allclose(calc[new], calc[old], rtol=1e-8, atol=1e-10):
            raise ValueError(f'Recalculated values do not match the submitted source data: {new}')
    cond = calc.groupby(['Study_ID','Condition_ID','Condition_Description'],as_index=False).agg(
        Number_of_Size_Bins=('Original_Size_Bin','size'),
        Particle_Abundance_items_per_kg=('Particle_Abundance_items_per_kg','sum'),
        Morphology_Corrected_Mass_Concentration_mg_per_kg=('Recalculated_Total_mg_per_kg','sum'))
    expected = cond_ref[['Study_ID','Condition_ID','Morphology_Corrected_Mass_Concentration_mg_per_kg']]
    chk = cond.merge(expected,on=['Study_ID','Condition_ID'],suffixes=('_calc','_source'))
    if not np.allclose(chk['Morphology_Corrected_Mass_Concentration_mg_per_kg_calc'],chk['Morphology_Corrected_Mass_Concentration_mg_per_kg_source'],rtol=1e-8,atol=1e-10):
        raise ValueError('Condition-level recalculated values do not match Dataset S3.')
    study = cond.groupby('Study_ID')['Morphology_Corrected_Mass_Concentration_mg_per_kg'].median().reset_index(name='Study_Median_mg_per_kg')
    study = study.sort_values('Study_ID')
    expected_study = study_ref[['Study_ID','Study_Median_Mass_Concentration_mg_per_kg']]
    chk2=study.merge(expected_study,on='Study_ID')
    if not np.allclose(chk2['Study_Median_mg_per_kg'],chk2['Study_Median_Mass_Concentration_mg_per_kg'],rtol=1e-8,atol=1e-10):
        raise ValueError('Study medians do not match Dataset S3.')
    q = np.percentile(study['Study_Median_mg_per_kg'],[10,50,90],method='linear')
    result=pd.DataFrame({'Statistic':['P10','P50','P90'],'Value_mg_per_kg_dry_soil':q})
    expected_q=stats_ref.set_index('Statistic')['Value'].to_dict()
    for k,v in zip(['P10','P50','P90'],q):
        if abs(float(v)-float(expected_q[k]))>1e-8: raise ValueError(f'{k} does not match Dataset S3.')
    calc.to_csv(out/'Fig3b_Particle_Bin_Recalculated.csv',index=False)
    cond.to_csv(out/'Fig3b_Condition_Level_Estimates.csv',index=False)
    study.to_csv(out/'Fig3b_Study_Level_Medians.csv',index=False)
    result.to_csv(out/'Fig3b_Field_Reference_Statistics.csv',index=False)
    print(result.to_string(index=False))
    print(f'Validated {len(calc)} particle-size bins, {len(cond)} conditions, and {len(study)} studies.')
if __name__ == '__main__': main()
