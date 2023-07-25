import datarobot as dr

dr.Client(config_path='drconfig.yaml')

import pandas as pd
from datetime import date

# trainingData = pd.read_csv('./winequality-white-training.csv')
# trainingData.head()
#
#
#
# projectName = 'Pyhton Wine Quality ' + dat%e.today().strftime(format='m-%d-%Y')
# print(projectName)
#
# project = dr.Project.create(
#     sourcedata=trainingData,
#     project_name=projectName
# )
#
# print(project.id, project.project_name)
#
# project.get_features()
#
# project.set_target(target='quality')
#
# project.wait_for_autopilot()

# # Viewing the Model

for p in dr.Project.list():
    print(p.id, p.project_name)

projectId = '64af92710719c2ee6f0d815c'
project = dr.Project.get(projectId)
print(project)

models = project.get_models()
for m in models:
    print(m.id, m.model_type)

recommendedModel = dr.ModelRecommendation.get(project.id).get_model()
print(recommendedModel, recommendedModel.model_type)

# # Deploying the Model

deployment = dr.Deployment.create_from_learning_model(
    model_id=recommendedModel.id,
    label='Wine Quality',
    description='Model for scoring wine quality'
)

# # Predictions

job = dr.BatchPredictionJob.score(
    deployment=deployment.id,
    passthrough_columns=['wine_id'],
    intake_settings={
        'type': 'localFile',
        'file': './winequality-white-score.csv'
    },
    output_settings={
        'type': 'localFile',
        'path': './winequality-white-predictions.csv'
    }
)
