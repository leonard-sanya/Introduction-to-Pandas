# -*- coding: utf-8 -*-
"""
What is pandas? \\


*   Python library used for working with data sets.
*   It has functions for analyzing, cleaning, exploring, and manipulating data.

Key Features of Pandas:


*   Fast and efficient DataFrame object with default and customized indexing.
*   Tools for loading data into in-memory data objects from different file formats.
*   Data alignment and integrated handling of missing data.
*   Reshaping and pivoting of date sets.
*   Label-based slicing, indexing and subsetting of large data sets.
*   Columns from a data structure can be deleted or inserted.
*   Group by data for aggregation and transformations.
*   High performance merging and joining of data.

To use pandas you need to install in local machine :
`
!pip install pandas
`
"""

# Installation of pandas
#!pip install pandas

"""alias (as) : In Python alias are an alternate name for referring to the same thing."""



from google.colab import drive
drive.mount('/content/drive')

# Import pandas as pd
import pandas as pd

"""In some project you have to use an specifique version of your library .  to check the Pandas Version : ```
pd.__version__ ```
"""

pd.__version__

"""# Introducing Pandas Objects

At the very basic level, Pandas objects can be thought of as enhanced versions of NumPy structured arrays in which the rows and columns are identified with labels rather than simple integer indices.
As we will see during the course of this chapter, Pandas provides a host of useful tools, methods, and functionality on top of the basic data structures, but nearly everything that follows will require an understanding of what these structures are.
Thus, before we go any further, let's introduce these two fundamental Pandas data structures: the ``Series``, ``DataFrame``.

We will start our code sessions with the standard NumPy and Pandas imports:
"""

import numpy as np
import pandas as pd

"""## The Pandas Series Object

A Pandas ``Series`` is a one-dimensional array of indexed data.
It can be created from a list or array as follows:
"""

data = pd.Series([0.25, 0.5, 0.75, 1.0])
data

"""As we see in the output, the ``Series`` wraps both a sequence of values and a sequence of indices, which we can access with the ``values`` and ``index`` attributes.
The ``values`` are simply a familiar NumPy array:
"""

data.values

"""The ``index`` is an array-like object of type ``pd.Index``, which we'll discuss in more detail momentarily."""

data.index

"""Like with a NumPy array, data can be accessed by the associated index via the familiar Python square-bracket notation:"""

data[1]

data[1:3]

"""As we will see, though, the Pandas ``Series`` is much more general and flexible than the one-dimensional NumPy array that it emulates.

### ``Series`` as generalized NumPy array

From what we've seen so far, it may look like the ``Series`` object is basically interchangeable with a one-dimensional NumPy array.
The essential difference is the presence of the index: while the Numpy Array has an *implicitly defined* integer index used to access the values, the Pandas ``Series`` has an *explicitly defined* index associated with the values.

This explicit index definition gives the ``Series`` object additional capabilities. For example, the index does not need to be an integer, but can consist of values of any desired type.
For example, if we wish, we can use strings as an index:
"""

data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
data

"""And the item access works as expected:"""

data['b']

"""We can even use non-contiguous or non-sequential indices:"""

data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=[2, 3, 3, 7])
data

data[3]

"""### Series as specialized dictionary

The ``Series``-as-dictionary analogy can be made even more clear by constructing a ``Series`` object directly from a Python dictionary:
"""

population_dict = {'Senegal': 38332521,
                   'Cameroun': 26448193,
                   'Mali': 19651127,
                   'Sudan': 19552860,
                   'Ghana': 12882135}
population = pd.Series(population_dict)
population

"""By default, a ``Series`` will be created where the index is drawn from the keys.
From here, typical dictionary-style item access can be performed:
"""

population['Senegal']

"""Unlike a dictionary, though, the ``Series`` also supports array-style operations such as slicing:"""

population['Cameroun':'Sudan']

"""### Constructing Series objects

We've already seen a few ways of constructing a Pandas ``Series`` from scratch; all of them are some version of the following:

```python
>>> pd.Series(data, index=index)
```

where ``index`` is an optional argument, and ``data`` can be one of many entities.

For example, ``data`` can be a list or NumPy array, in which case ``index`` defaults to an integer sequence:
"""

pd.Series([2, 4, 6])

"""``data`` can be a scalar, which is repeated to fill the specified index:"""

pd.Series(5, index=[100, 200, 300])

"""``data`` can be a dictionary, in which ``index`` defaults to the sorted dictionary keys:"""

pd.Series({2:'a', 1:'b', 3:'c'})

"""In each case, the index can be explicitly set if a different result is preferred:"""

pd.Series({2:'a', 1:'b', 3:'c'}, index=[3, 2])

"""Notice that in this case, the ``Series`` is populated only with the explicitly identified keys.

### The Pandas DataFrame Object

The next fundamental structure in Pandas is the ``DataFrame``.
Like the ``Series`` object discussed in the previous section, the ``DataFrame`` can be thought of either as a generalization of a NumPy array, or as a specialization of a Python dictionary.
We'll now take a look at each of these perspectives.

### DataFrame as a generalized NumPy array
If a ``Series`` is an analog of a one-dimensional array with flexible indices, a ``DataFrame`` is an analog of a two-dimensional array with both flexible row indices and flexible column names.
Just as you might think of a two-dimensional array as an ordered sequence of aligned one-dimensional columns, you can think of a ``DataFrame`` as a sequence of aligned ``Series`` objects.
Here, by "aligned" we mean that they share the same index.

To demonstrate this, let's first construct a new ``Series`` listing the area of each of the five states discussed in the previous section:
"""

area_dict = {'Senegal': 423967, 'Cameroun': 695662, 'Mali': 141297,
             'Sudan': 170312, 'Ghana': 149995}
area = pd.Series(area_dict)
area

"""Now that we have this along with the ``population`` Series from before, we can use a dictionary to construct a single two-dimensional object containing this information:"""

states = pd.DataFrame({'population': population,
                       'area': area})
states

"""Like the ``Series`` object, the ``DataFrame`` has an ``index`` attribute that gives access to the index labels:"""

states.index

"""Additionally, the ``DataFrame`` has a ``columns`` attribute, which is an ``Index`` object holding the column labels:"""

states.columns

"""the ``DataFrame`` can be thought of as a generalization of a two-dimensional NumPy array, where both the rows and columns have a generalized index for accessing the data.

### DataFrame as specialized dictionary

Similarly, we can also think of a ``DataFrame`` as a specialization of a dictionary.
Where a dictionary maps a key to a value, a ``DataFrame`` maps a column name to a ``Series`` of column data.
For example, asking for the ``'area'`` attribute returns the ``Series`` object containing the areas we saw earlier:
"""

states['area']

"""### Constructing DataFrame objects

A Pandas ``DataFrame`` can be constructed in a variety of ways.
Here we'll give several examples.

#### From a single Series object

A ``DataFrame`` is a collection of ``Series`` objects, and a single-column ``DataFrame`` can be constructed from a single ``Series``:
"""

pd.DataFrame(population, columns=['population'])

"""#### From a list of dicts

Any list of dictionaries can be made into a ``DataFrame``.
We'll use a simple list comprehension to create some data:
"""

data = [{'a': i, 'b': 2 * i} for i in range(3)]
pd.DataFrame(data)

"""Even if some keys in the dictionary are missing, Pandas will fill them in with ``NaN`` (i.e., "not a number") values:"""

t =pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4},{'b': 3, 'c': 4}])
t

t.index

"""#### From a dictionary of Series objects

As we saw before, a ``DataFrame`` can be constructed from a dictionary of ``Series`` objects as well:
"""

pd.DataFrame({'population': population,'area': area})

"""#### From a two-dimensional NumPy array

Given a two-dimensional array of data, we can create a ``DataFrame`` with any specified column and index names.
If omitted, an integer index will be used for each:
"""

d = pd.DataFrame(data=[[2,3],[4,5],[6,7]],
             index=['a', 'a', 'c'])
d

pd.DataFrame(np.random.rand(3, 2),
             columns=['foo', 'bar'],
             index=['b', 'a', 'c'])

"""### Reading data files

Data can be stored in any of a number of different forms and formats. By far the most basic of these is the humble CSV file. When you open a CSV file you get something that looks like this:

```
Product A,Product B,Product C,
30,21,9,
35,34,1,
41,11,11
```

So a CSV file is a table of values separated by commas. Hence the name: "Comma-Separated Values", or CSV.

Let's now set aside our toy datasets and see what a real dataset looks like when we read it into a DataFrame. We'll use the `pd.read_csv()` function to read the data into a DataFrame. This goes thusly: `data = pd.read_csv("filename.csv")`
"""

import pandas as pd

data = pd.read_csv("/content/sample_data/california_housing_train.csv")

"""We can use the `shape` attribute to check how large the resulting DataFrame is:"""

data.shape

"""So our new DataFrame has 17,000 records split across 9 different columns.

We can examine the contents of the resultant DataFrame using the `head()` command, which grabs the first five rows:
"""

data.head()

"""How we can save the file in our local machine ??

*** Exercice*** 😊
"""

data.to_csv("whatever.csv",index=False)

"""# Indexing, Selecting & Assigning

Selecting specific values of a pandas DataFrame or Series to work on is an implicit step in almost any data operation you'll run, so one of the first things you need to learn in working with data in Python is how to go about selecting the data points relevant to you quickly and effectively.

## Native accessors

Native Python objects provide  good ways of indexing data. Pandas carries all of these over, which helps make it easy to start with.

Consider this DataFrame:
"""

import pandas as pd

data = pd.read_csv("/content/sample_data/california_housing_train.csv")
data.head()

"""In Python, we can access the property of an object by accessing it as an attribute. A `book` object, for example, might have a `title` property, which we can access by calling `book.title`. Columns in a pandas DataFrame work in much the same way.

Hence to access the `median_house_value` property of `data` we can use:
"""

data.median_house_value

"""If we have a Python dictionary, we can access its values using the indexing (`[]`) operator. We can do the same with columns in a DataFrame:"""

data['median_house_value']

"""These are the two ways of selecting a specific Series out of a DataFrame.

Doesn't a pandas Series look kind of like a fancy dictionary? It pretty much is, so it's no surprise that, to drill down to a single specific value, we need only use the indexing operator `[]` once more:
"""

data['median_house_value'][0]

"""## Indexing in pandas

The indexing operator and attribute selection are nice because they work just like they do in the rest of the Python ecosystem. As a novice, this makes them easy to pick up and use. However, pandas has its own accessor operators, `loc` and `iloc`. For more advanced operations, these are the ones you're supposed to be using.

### Index-based selection

Pandas indexing works in one of two paradigms. The first is **index-based selection**: selecting data based on its numerical position in the data. `iloc` follows this paradigm.

To select the first row of data in a DataFrame, we may use the following:
"""

data.iloc[0]

"""Both `loc` and `iloc` are row-first, column-second. This is the opposite of what we do in native Python, which is column-first, row-second.

This means that it's marginally easier to retrieve rows, and marginally harder to get retrieve columns. To get a column with `iloc`, we can do the following:
"""

data.iloc[:,0]

"""On its own, the `:` operator, which also comes from native Python, means "everything". When combined with other selectors, however, it can be used to indicate a range of values. For example, to select the `longitude` column from just the first, second, and third row, we would do:"""

data.iloc[:3,0]

"""Or, to select just the second and third entries, we would do:"""

data.iloc[1:3, 0]

"""It's also possible to pass a list:"""

data.iloc[[0, 1, 2], 0]

"""Finally, it's worth knowing that negative numbers can be used in selection. This will start counting forwards from the _end_ of the values. So for example here are the last five elements of the dataset."""

data.iloc[-5:]

"""### Label-based selection

The second paradigm for attribute selection is the one followed by the `loc` operator: **label-based selection**. In this paradigm, it's the data index value, not its position, which matters.

For example, to get the first entry in `data`, we would now do the following:
"""

data.loc[0 , "longitude"]

data.loc[:,["latitude","longitude"]]

"""## Conditional selection

So far we've been indexing various strides of data, using structural properties of the DataFrame itself. To do *interesting* things with the data, however, we often need to ask questions based on conditions.

For example, suppose that we're interested specifically in better-than-average wines produced in Italy.

We can start by checking if each wine is Italian or not:
"""

data.latitude==34.40

"""This operation produced a Series of `True`/`False` booleans based on the `latitude` of each record.  This result can then be used inside of `loc` to select the relevant data:"""

data.loc[data.latitude == 34.40]

"""We can use the ampersand (`&`) to bring the two questions together:"""

data.loc[(data.latitude == 34.40) & (data.total_rooms >= 7650.0)]

data.loc[(data.latitude == 34.40) | (data.total_rooms >= 7650.0)]

"""# Summary Functions and Mapping"""

import pandas as pd

data = pd.read_csv("/content/sample_data/california_housing_train.csv")
data

"""## Summary functions

Pandas provides many simple "summary functions" (not an official name) which restructure the data in some useful way. For example, consider the `describe()` method:
"""

data.describe()

"""This method generates a high-level summary of the attributes of the given column. It is type-aware, meaning that its output changes based on the data type of the input. The output above only makes sense for numerical data;

If you want to get some particular simple summary statistic about a column in a DataFrame or a Series, there is usually a helpful pandas function that makes it happen.

For example, to see the mean of the points allotted (e.g. how is the latitude in average), we can use the `mean()` function:
"""

data.latitude.mean()

"""To see a list of unique values we can use the `unique()` function:"""

data.latitude.unique()

"""To see a list of unique values _and_ how often they occur in the dataset, we can use the `value_counts()` method:"""

data.latitude.value_counts()

"""## Mapping functions

A **map** is a term, borrowed from mathematics, for a function that takes one set of values and "maps" them to another set of values. In data science we often have a need for creating new representations from existing data, or for transforming data from the format it is in now to the format that we want it to be in later. Maps are what handle this work, making them extremely important for getting your work done!

There are two mapping methods that you will use often.

[`map()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.map.html) is the first, and slightly simpler one. It's used only on `Series`. For example, suppose that we wanted to remean the latitude to 0. We can do this as follows:
"""

latitude_mean = data.latitude.mean()
data.latitude.map(lambda p: p - latitude_mean)

def remean_latitude(row):
  latitude_mean = 4
  row1 = row - latitude_mean
  return row, row1

data.latitude.map(lambda x : remean_latitude(x))



"""We also `apply()` can be used on both `Series`(same as `map`) and `DataFrame`. When used on `DataFrame`, it applies on every column of the dataframe."""

data.apply(lambda x: x/10)

data.latitude.apply(lambda x: x/10)

"""Lastly, we have `applymap()`, used only on dataFrame. It's the same as `apply()` function on `DataFrame`."""

data.applymap(lambda x: x/10)

"""# Grouping and Sorting

Maps allow us to transform data in a DataFrame or Series one value at a time for an entire column. However, often we want to group our data, and then do something specific to the group the data is in.

As you'll learn, we do this with the `groupby()` operation.  We'll also cover some additional topics, such as more complex ways to index your DataFrames, along with how to sort your data.

One function we've been using heavily thus far is the `value_counts()` function. We can replicate what `value_counts()` does by doing the following:
"""

data.groupby('latitude').longitude.count()

"""`groupby()` created a group of data which allotted the same point values to the given wines. Then, for each of these groups, we grabbed the `latitude` column and counted how many times it appeared.  `value_counts()` is just a shortcut to this `groupby()` operation.

We can use any of the summary functions we've used before with this data. For example, to get the minimum longitude in each latitude value point, we can do the following:
"""

# data.groupby('latitude').longitude

data.groupby('latitude').longitude.min()

"""You can think of each group we generate as being a slice of our DataFrame containing only data with values that match. This DataFrame is accessible to us directly using the `apply()` method, and we can then manipulate the data in any way we see fit. For example, here's one way of selecting the two first longitude from each latitude in the dataset:"""

data.groupby('latitude').apply(lambda df: df.longitude.iloc[0:2])

data

dfs = pd.DataFrame()
dfs["country"] = ["togo", "Senegal", "sudan", "tunisia"]
dfs["region"] = ['w', 'w', 'n', 's']
dfs["pop"] = [12,21,23,43]
dfs

dfs.groupby('region').apply(lambda df: df.region.iloc[0:2])

"""Another `groupby()` method worth mentioning is `agg()`, which lets you run a bunch of different functions on your DataFrame simultaneously. For example, we can generate a simple statistical summary of the dataset as follows:"""

data.groupby(['latitude']).latitude.agg([len, min, max])

"""## Sorting


To get data in the order want it in, we can sort it ourselves.  The `sort_values()` method is handy for this.
"""

latitude_data = data.reset_index()
latitude_data

latitude_data.sort_values(by="latitude")

"""`sort_values()` defaults to an ascending sort, where the lowest values go first. However, most of the time we want a descending sort, where the higher numbers go first. That goes thusly:"""

latitude_data.sort_values(by='latitude', ascending=False)

"""To sort by index values, use the companion method `sort_index()`. This method has the same arguments and default order:"""

data.sort_index()

"""Finally, know that you can sort by more than one column at a time:

# Data Types and Missing Values
The data type for a column in a DataFrame or a Series is known as the dtype.

You can use the ``dtype`` property to grab the type of a specific column. For instance, we can get the dtype of the population column in the reviews DataFrame:
"""

data.population.dtype

"""Alternatively, the ``dtypes`` property returns the dtype of every column in the DataFrame:"""

data.dtypes

"""It's possible to convert a column of one type into another wherever such a conversion makes sense by using the ``astype()`` function. For example, we may transform the points column from its existing ``float64`` data type into a ``int64`` data type:"""

data.population.astype('int64')

"""A DataFrame or Series index has its own dtype, too:"""

data.index.dtype

"""### Detecting null values
Pandas data structures have two useful methods for detecting null data: ``isnull()`` and ``notnull()``.
Either one will return a Boolean mask over the data. For example:
"""

data = pd.Series([1, np.nan, 'hello', None])
data

data.isnull()

data[data.notnull()]

"""The ``isnull()`` and ``notnull()`` methods produce similar Boolean results for ``DataFrame``s.

### Dropping null values

In addition to the masking used before, there are the convenience methods, ``dropna()``
(which removes NA values) and ``fillna()`` (which fills in NA values). For a ``Series``,
the result is straightforward:
"""

data.dropna()

"""For a ``DataFrame``, there are more options.
Consider the following ``DataFrame``:
"""

df = pd.DataFrame([[1,      np.nan, 2],
                   [2,      3,      5],
                   [np.nan, 4,      6]])
df

"""We cannot drop single values from a ``DataFrame``; we can only drop full rows or full columns.
Depending on the application, you might want one or the other, so ``dropna()`` gives a number of options for a ``DataFrame``.

By default, ``dropna()`` will drop all rows in which *any* null value is present:
"""

df.dropna()

"""Alternatively, you can drop NA values along a different axis; ``axis=1`` drops all columns containing a null value:"""

df.dropna(axis='columns')

"""But this drops some good data as well; you might rather be interested in dropping rows or columns with *all* NA values, or a majority of NA values.
This can be specified through the ``how`` or ``thresh`` parameters, which allow fine control of the number of nulls to allow through.

The default is ``how='any'``, such that any row or column (depending on the ``axis`` keyword) containing a null value will be dropped.
You can also specify ``how='all'``, which will only drop rows/columns that are *all* null values:
"""

df[6] = np.nan
df

df.dropna(axis='columns', how='all')

"""For finer-grained control, the ``thresh`` parameter lets you specify a minimum number of non-null values for the row/column to be kept:"""

df.dropna(axis='rows', thresh=2)

"""Here the first and last row have been dropped, because they contain only two non-null values.

### Filling null values

Sometimes rather than dropping NA values, you'd rather replace them with a valid value.
This value might be a single number like zero, or it might be some sort of imputation or interpolation from the good values.
You could do this in-place using the ``isnull()`` method as a mask, but because it is such a common operation Pandas provides the ``fillna()`` method, which returns a copy of the array with the null values replaced.

Consider the following ``Series``:
"""

data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
data

"""We can fill NA entries with a single value, such as zero:"""

data.fillna(0)

"""We can specify a forward-fill to propagate the previous value forward:"""

# forward-fill
data.fillna(method='ffill')

"""Or we can specify a back-fill to propagate the next values backward:"""

# back-fill
data.fillna(method='bfill')

"""For ``DataFrame``s, the options are similar, but we can also specify an ``axis`` along which the fills take place:"""

df

df.fillna(method='ffill', axis=1)

