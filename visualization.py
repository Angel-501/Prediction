from pandas import *
from numpy import *
from matplotlib.pyplot import *
from seaborn import *
from scipy.stats import *

incidents= read_csv("G://Project//Incidents_service.csv")
incidents.isnull().sum()
incidents.describe()

heatmap(incidents.isnull(),yticklabels=False,cbar=True,cmap='viridis')


incidents.ID.value_counts()
#incidents.ID.value_counts().plot(kind="pie")
incidents.ID_status.value_counts()
incidents.ID_status.value_counts().plot(kind="boxplot")
incidents.active.value_counts()
incidents.active.value_counts().plot(kind="pie")
incidents.count_reassign.value_counts()
incidents.count_reassign.value_counts().plot(kind="pie")
incidents.count_opening.value_counts()
incidents.count_opening.value_counts().plot(kind="pie")
incidents.count_updated.value_counts()
incidents.count_updated.value_counts().plot(kind="pie")
incidents.ID_caller.value_counts()
incidents.ID_caller.value_counts().plot(kind="bar")
incidents.opened_by.value_counts()
incidents.opened_by.value_counts().plot(kind="pie")
incidents.opened_time.value_counts()
incidents.opened_time.value_counts().plot(kind="pie")
incidents.Created_by.value_counts()
#incidents.Created_by.value_counts().plot(kind="pie")
incidents.created_at.value_counts()
#incidents.created_at.value_counts().plot(kind="pie")
incidents.updated_by.value_counts()
incidents.updated_by.value_counts().plot(kind="pie")
incidents.updated_at.value_counts()
incidents.updated_at.value_counts().plot(kind="pie")
incidents.type_contact.value_counts()
incidents.type_contact.value_counts().plot(kind="pie")
incidents.location.value_counts()
incidents.location.value_counts().plot(kind="pie")
incidents.category_ID.value_counts()
incidents.category_ID.value_counts().plot(kind="pie")
incidents.user_symptom.value_counts()
incidents.user_symptom.value_counts().plot(kind="pie")
incidents.impact.value_counts()
incidents.impact.value_counts().plot(kind="pie")
incidents.Support_group.value_counts()
incidents.Support_group.value_counts().plot(kind="pie")
incidents.support_incharge.value_counts()
incidents.support_incharge.value_counts().plot(kind="pie")
incidents.Doc_knowledge.value_counts()
incidents.Doc_knowledge.value_counts().plot(kind="pie")
incidents.confirmation_check.value_counts()
incidents.confirmation_check.value_counts().plot(kind="pie")
incidents.notify.value_counts()
incidents.notify.value_counts().plot(kind="pie")



incidents.loc[45:50,["count_reassign","count_opening","count_updated"]]

#Skewness and kurtosis
incidents.skew()
incidents.kurt()

boxplot(incidents['ID'])
boxplot(incidents['count_reassign'])

#Histogram

hist(incidents['ID_status'])
hist(incidents['active'])
hist(incidents['count_reassign'])
hist(incidents['count_opening'])
hist(incidents['count_updated'])
hist(incidents['ID_caller'])
hist(incidents['opened_by'])
hist(incidents['opened_time'])
hist(incidents['Created_by'])
hist(incidents['created_at'])
hist(incidents['updated_by'])
hist(incidents['updated_at'])
hist(incidents['type_contact'])
hist(incidents['location'])
hist(incidents['category_ID'])
hist(incidents['user_symptom'])
hist(incidents['impact'])
hist(incidents['Support_group'])
hist(incidents['support_incharge'])
hist(incidents['Doc_knowledge'])
hist(incidents['confirmation_check'])
hist(incidents['notify'])
hist(incidents['problem_id'])
hist(incidents['change_request'])

pairplot(incidents.iloc[:3:5])

plot(arange(32),incidents.impact)

incidents.iloc[:,:4]
incidents.corr()


sb.boxplot(x="ID_status",y="impact",data=incidents,palette = "hls")
countplot(x="ID_status", data=incidents)
crosstab(incidents.impact,incidents.ID_status).plot(kind="bar")

countplot(x="active", data=incidents)
crosstab(incidents.impact,incidents.active).plot(kind="bar")

countplot(x="count_reassign", data=incidents)
crosstab(incidents.impact,incidents.count_reassign).plot(kind="bar")

countplot(x="count_opening", data=incidents)
crosstab(incidents.impact,incidents.count_opening).plot(kind="bar")

countplot(x="count_updated", data=incidents)
crosstab(incidents.impact,incidents.count_updated).plot(kind="bar")

countplot(x="ID_caller", data=incidents)
crosstab(incidents.impact,incidents.ID_caller).plot(kind="bar")

countplot(x="type_contact", data=incidents)
crosstab(incidents.impact,incidents.type_contact).plot(kind="bar")

countplot(x="notify", data=incidents)
crosstab(incidents.impact,incidents.notify).plot(kind="bar")

countplot(x="problem_id", data=incidents)
crosstab(incidents.impact,incidents.problem_id).plot(kind="bar")

countplot(x="confirmation_check", data=incidents)
crosstab(incidents.impact,incidents.confirmation_check).plot(kind="bar")

countplot(x="Doc_knowledge", data=incidents)
crosstab(incidents.impact,incidents.Doc_knowledge).plot(kind="bar")

countplot(x="support_incharge", data=incidents)
crosstab(incidents.impact,incidents.support_incharge).plot(kind="bar")

countplot(x="Support_group", data=incidents)
crosstab(incidents.impact,incidents.Support_group).plot(kind="bar")

countplot(x="user_symptom", data=incidents)
crosstab(incidents.impact,incidents.user_symptom).plot(kind="bar")

countplot(x="category_ID", data=incidents)
crosstab(incidents.impact,incidents.category_ID).plot(kind="bar")

countplot(x="location", data=incidents)
crosstab(incidents.impact,incidents.location).plot(kind="bar")

boxplot(x="ID_status",y="impact",data=incidents,palette = "hls")

incidents.corr(method='pearson', min_periods=1)
incidents.shape



