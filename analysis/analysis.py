
#%%
import pandas as pd 
import wandb
api = wandb.Api()

# Project is specified by <entity/project-name>
def load_data_sweeps(sweep_name='6s1hz3bl'):
    runs = api.runs("cairi/OVANET_DMT".format(sweep_name))

    summary_list, config_list, name_list = [], [], []
    sweep_list = []
    for run in runs: 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict)
        try:
            sweep_list.append(run.sweep.id)
        except:
            sweep_list.append('none')

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    # runs_df = pd.DataFrame({
    #     "summary": summary_list,
    #     "config": config_list,
    #     "name": name_list
    #     })

    summary_list = pd.DataFrame().from_dict(summary_list).drop('lr', axis=1)

    data = pd.concat(
        [
            pd.DataFrame().from_dict(config_list),
            summary_list,
            pd.DataFrame(name_list),
            pd.DataFrame(sweep_list, columns=['sweep']),
        ],
        axis=1
    )

    return data[data['sweep'] == sweep_name], runs._sweeps[sweep_name]
# %%

data, sweep_info = load_data_sweeps()
data

# %%

parm = sweep_info.config['parameters']

gird_search_parm_list = []

for k1 in parm.keys():
    # print(k1)
    if len(parm[k1]['values']) > 1:
        print('-------------')
        gird_search_parm_list.append(k1)
        print(k1)
        print(len(parm[k1]['values']))
        # print(parm[k1]['values'])
    # if type(parm[k1]) != str:
    #     for k2 in parm[k1].keys():
    #         print('-->',k2)

# gird_search_parm_list.remove('lr')
gird_search_parm_list.remove('source_data')
gird_search_parm_list.remove('target_data')
print(gird_search_parm_list)
# %%
import itertools

item_tool = []
for item in gird_search_parm_list:
    item_tool.append(list(set(data[item].to_list())))



data_group = data.groupby(gird_search_parm_list).mean()
data_group['count'] = data.groupby(gird_search_parm_list).size()
data_group = data_group[['h_score', 'h_score_epoch', 'count']]

data_group.sort_values(['h_score_epoch'], ascending=False)
# a = data.set_index(gird_search_parm_list)
# df_result = None
# dict_item = {}
# for item in itertools.product(*item_tool):
#     try:
#         z = a.loc[item]
#         df_item = pd.DataFrame(
#             z[['h_score', 'h_score_epoch']].mean(),
#             columns=[item],
#             ).T
#         df_item['num'] = z.shape[0]

#         if df_result is None:
#             df_result = df_item
#         else:
#             df_result = pd.concat([df_result, df_item])
#         print(z.shape)
#     except:
#         print('---')

# df_result.sort_values(['h_score_epoch'], ascending=False)
# data
# %%
