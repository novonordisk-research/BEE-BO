import pandas as pd
import json


runs = {
    'ackley2_q100' : 0.0,
    'levy2_q100' : 0.0,
    'rastrigin2_q100': 0.0,
    'rosenbrock2_q100': 0.0,
    'styblinskitang2_q100': 39.166166 *2,
    'shekel4_q100': 10.5363,
    'hartmann6_q100': 3.32237,
    'cosine8_q100': 0.8, 

    'ackley10_q100': 0.0,
    'levy10_q100': 0.0,
    'powell10_q100' : 0.0,
    'rastrigin10_q100': 0.0,
    'rosenbrock10_q100': 0.0,
    'styblinskitang10_q100': 39.166166 * 10,

    'ackley20_q100': 0.0,
    'levy20_q100': 0.0,
    'powell20_q100': 0.0,
    'rastrigin20_q100': 0.0,
    'rosenbrock20_q100': 0.0,
    'styblinskitang20_q100': 39.166166 * 20,

    'ackley50_q100': 0.0,
    'levy50_q100': 0.0,
    'powell50_q100' : 0.0,
    'rastrigin50_q100': 0.0,
    'rosenbrock50_q100': 0.0,
    'styblinskitang50_q100': 39.166166 * 50,

    'embeddedhartmann100_q100': 3.32237,
    'ackley100_q100': 0.0,
    'levy100_q100': 0.0,
    'powell100_q100' : 0.0,
    'rastrigin100_q100': 0.0,
    'rosenbrock100_q100': 0.0,
    'styblinskitang100_q100': 39.166166 * 100,
}



def get_runs(run_name, k=0.1):
    all_timings = []
    for i in [0,1,2,3,4]:
        files = [
            f'./runs_nomorebeta_{i}/{run_name}/beebo_explore_parameter{k/2}/config.json',
            f'./runs_nomorebeta_{i}/{run_name}/maxbeebo_explore_parameter{k/2}/config.json',
            f'./runs_{i}/{run_name}/qucb_explore_parameter{k}/config.json',
            f'./runs_{i}/{run_name}/qei/config.json',
            f'./runs_{i}/{run_name}/thompson/config.json',
            f'./runs_{i}/{run_name}/kriging_believer/config.json',
            f'./runs_{i}/{run_name}/gibbon/config.json',
            f'./runs_{i}/{run_name}/modifiedgibbon/config.json',
            f'./runs_{i}/{run_name}/random/config.json',
        ]

        for f in files:
            data = json.load(open(f))
            run_time = data['run_time']

            all_timings.append({
                'file': f,
                'run': run_name,
                'run_time': run_time
            })

    return pd.DataFrame(all_timings)




all_results = []
for k in [0.1, 1.0, 10.0]:

    # get the runs.
    for run in runs:
        df = get_runs(run, k=k)

        # df['run'] = run
        df['algo'] = df['file'].apply(lambda x: x.split('/')[3])
        df['rep'] = df['file'].apply(lambda x: x.split('/')[1])

        del df['file']
        
        all_results.append(df)

df = pd.concat(all_results)

print('Total run time [h]:', df['run_time'].sum() / 3600)
print('Total per algo')
print(df.groupby('algo')['run_time'].sum() / 3600)

df_agg = df.groupby(['algo', 'run'])['run_time'].agg(['mean', 'std'])
df_agg.to_csv('timings.csv')


