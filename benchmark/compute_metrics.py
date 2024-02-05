import pandas as pd
import numpy as np


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



def get_runs(run_name, k=0.1, suffix=''):
    all_dfs = []
    for i in [0,]:#1,2,3,4]:
        files = [
            f'./runs_{i}/{run_name}/beebo_explore_parameter{k/2}/experiment_log{suffix}.csv',
            f'./runs_{i}/{run_name}/qucb_explore_parameter{k}/experiment_log{suffix}.csv',
            f'./runs_{i}/{run_name}/qei/experiment_log.csv',
            f'./runs_{i}/{run_name}/thompson/experiment_log.csv',
            f'./runs_{i}/{run_name}/random/experiment_log.csv',
        ]

        for f in files:
            df = pd.read_csv(f)
            df['file'] = f.split('/')[-2]
            df['seed'] = i
            all_dfs.append(df)
    df = pd.concat(all_dfs)
    return df





for k in [0.1, 1.0, 10.0]:

    ###
    ### Batch regret
    ###



    # get the runs.
    all_results = []
    for run in runs:
        df = get_runs(run, k=k, suffix='_round10_full_exploit')
        df['optimum'] = runs[run]

        df['regret'] = df['y'] - df['optimum']
        df['run'] = run
        
        all_results.append(df)


    df = pd.concat(all_results)
    df = df.loc[df['round']==10]
    df = df.groupby(['file', 'run'])['regret'].agg(['sum']).reset_index() # batch regret
    df['algo'] = df['file'].apply(lambda x: x.split('_')[0])

    df = df.pivot(index='run', columns='algo', values='sum').reset_index()
    df['dim'] = df['run'].str.split('(\d+)', expand=True)[1].astype(int)
    df['problem'] = df['run'].str.split('(\d+)', expand=True)[0]


    # normalize as % of random
    df_norm = df.copy()
    df_norm['beebo'] = df_norm['beebo'] / df_norm['random']
    df_norm['qucb'] = df_norm['qucb'] / df_norm['random']
    df_norm['qei'] = df_norm['qei'] / df_norm['random']
    df_norm['thompson'] = df_norm['thompson'] / df_norm['random']
    df_norm['random'] = df_norm['random'] / df_norm['random'] # sanity check


    df_norm.sort_values(['dim', 'problem'])[['dim', 'problem', 'beebo', 'qucb', 'qei', 'thompson', 'random']].to_csv(f'batch_regret_{k}.csv', index=False)



    ###
    ### Best so far
    ###

    # get the runs.
    all_results = []
    for run in runs:
        df = get_runs(run, k=k, suffix='_round10_full_exploit')

        best_so_far = df.groupby(['file', 'seed', 'round'])['y'].max().groupby(['file', 'seed']).cummax()#.cummax()['y']

        seed_max = df.loc[df['round']==0].groupby('seed')['y'].max()
        seed_max.name = 'minimum'


        df = pd.DataFrame(best_so_far).reset_index().rename(columns={'y': 'best_so_far'})

        df = df.merge(seed_max, on='seed')
        df['maximimum'] = runs[run]
        #min-max normalize.
        df['best_so_far_normalized'] = (df['best_so_far'] - df['minimum']) / (df['maximimum'] - df['minimum'])

        df = df.loc[df['round']==10]
        df_end = df.groupby('file')[['best_so_far', 'best_so_far_normalized']].agg(['mean', 'std'])#df.groupby('file')['best_so_far'].agg(['mean', 'std'])
        df_end['run'] = run
        all_results.append(df_end)


    df = pd.concat(all_results)
    # make text row mean+/-std
    df['text'] = df.apply(lambda x: f'{x["best_so_far_normalized", "mean"]:.3f}',axis=1)

    # reshape so that each row is a problem, qUCB and BOSS are columns
    df =  df.reset_index()
    df['algo'] = df['file'].apply(lambda x: x.split('_')[0])

    df = df.pivot(index='run', columns='algo', values='text')
    df = df.reset_index()

    df['dim'] = df['run'].str.split('(\d+)', expand=True)[1].astype(int)
    df['problem'] = df['run'].str.split('(\d+)', expand=True)[0]
    df = df.sort_values(['dim', 'problem'])
    df['dim'] = df['run'].str.split('(\d+)', expand=True)[1]
    df = df[['problem', 'dim',  'beebo', 'qucb', 'qei', 'thompson']]
    df.to_csv(f'best_so_far_{k}.csv', index=False)