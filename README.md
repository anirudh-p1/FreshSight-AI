# FreshSight-AI

FreshSight AI is an integrated, AI-powered computer vision system designed to monitor the
freshness of perishable produce in real time at the retail stage of the food supply chain. The
system uses machine learning, a convolutional neural network (CNN) model, to analyse the
visual conditions of fresh produce directly on the shelf, generating a freshness score for
every item, and triggering one of three automated actions depending on how close the item
is to the end of its shelf life. The three automated actions are: 1 – continued sale at full price,
2 – an AI optimised price reduction to accelerate purchase, and 3 – an automated alert sent
to registered food bank partners via an API, who can arrange collection within a time
window. The system integrates with the electronic barcodes for different products.

2 – Hardware and Components
FreshSight AI uses small, low power RGB cameras mounted above each shelf section. Each
camera captures continuous images of the produce below. Computing units housed in
compact processors mounted behind the shelving, run the CNN inference locally, which
reduces latency and the reliance on internet. Data is then sent to a central store server, and
optionally to a cloud dashboard, which is accessible to store managers, and food banks that
are partnered with the store. The hardware uses low-energy components, so it can be run all
the time.
3 – The Machine Learning Model
The core part of FreshSight AI is a CNN trained on a large dataset of produce images that
are labelled by freshness stage (fresh, moderate, near expiry, and spoiled). CNNs are a type
of deep learning model that are used in image recognition as they learn visual features such
as colour changes, texture differences, and shrivelling to correspond to freshness levels.
The model is with supervised learning, as it uses labelled training images, and it learns to
give back the labels. Once used, it can assess and analyse thousands of items per hour,
with a target accuracy above 92%, which is much better than human visual inspection.
The model gives a freshness score for each item, from 0 to 100. The thresholds are different
for each produce category. For example, strawberries trigger at 70, whilst root vegetables
trigger at 50. These thresholds do vary over time, as they are trained using more feedback.
4 – Dynamic Pricing and Food Bank Integration
When the CNN gives a freshness score below a threshold, FreshSight AI automatically
updates the electronic shelf label for that item with an optimised discount price. The discount
is calculated using a secondary predictive model that estimates the minimum reduction
needed to ensure the item sells before expiry, which is based on historical sales data for that

product. This prevents over-discounting, which reduces revenue for the firms, and under-
discounting, which produces waste.

When an item’s score falls below the threshold for foodbanks, the system sends an automated
notification to the registered local food bank organisations via a simple REST API. The food
bank can see the item type, quantity, and the collection window on their partner app. This
allows for the efficient distribution of surplus food.
