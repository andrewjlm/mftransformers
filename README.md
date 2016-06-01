# mftransformers

One of my favorite `scikit-learn` features is the `Pipeline` class.
Pipelines allow you to quickly combine multiple feature selection, classifiers
or estimators into a single unit. Using an example from the `scikit-learn`
documentation, we could set up a simple pipeline that performs an unsupervised
dimension reduction using PCA and then performs classification using logistic
regression:

```python
pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
```

By doing this we've essentially created a compound estimator that we can operate
on as a whole (instead of working our data through each estimator individually).
Continuing with the example from the `scikit-learn` documentation, we can use
a grid search to set the parameters for number of components and inverse
regularization strength and then fit the model:

```python
estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              logistic__C=Cs))
estimator.fit(X_digits, y_digits)
```

See the (excellent) `scikit-learn` documentation for more details.

While `scikit-learn` includes PCA it lacks other matrix factorization methods.
The purpose of this package is to implement those methods in a way that is
compatible with the `scikit-learn` pipeline system.
