Given:

- 227 CT scans, each scan has 30 slices
- annotations.csv with slice-level annotations

```
slice_id,    case_id,       ANY,IPH,IVH,SAH,SDH
ID_c821d342b,CID_2de9ccc9d2,0,  0,  0,  0,  0
```

ANY column: 1 means slice contains haemorrgage, 0 means doesn't

IPH,IVH,SAH,SDH: four subtypes of haemorrhages. Could have multiple at the same time

#### Evaluation:

Sensitivity and specificity (exists or not in the brain), 1 or 0. Subtypes do not
matter as of now.


#### Thoughts:

Data is given in a format I am not used to working, a bunch of nested folders. How
could I move it?

ImageFolder approach didn't work out haha, pytorch doesn't support .dcm. Will need to
create a custom DataSet object