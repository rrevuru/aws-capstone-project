# Data Preparation

In this phase, going to leverage Lakeformation, Lakeformation provides a single interface to manage security controls and ease of use of Glue ETL features.

# Lake Formation

ratings.csv: userId, movieId, rating, timestamp
tags.csv: userId, movieId, tag, timestamp
movies.csv: movieId, title, genres
links.csv: movieId, imdbId, tmdbId


### Data Transformation


```
datasource0 = glueContext.create_dynamic_frame.from_catalog(database = "movies", table_name = "ratings19", transformation_ctx = "datasource0")

```

### Apply Mapping

````
applymapping1 = ApplyMapping.apply(frame = datasource0, mappings = [("userid", "long", "userid", "long"), ("movieid", "long", "movieid", "long"), ("rating", "float", "rating", "float"), ("timestamp", "timestamp", "timestamp", "timestamp")], transformation_ctx = "applymapping1")
````

