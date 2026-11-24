# Experiment 6 : Market Basket Analysis (Association Rule Mining using Apriori Algorithm)

## Aim
To perform Market Basket Analysis using the Apriori algorithm to identify frequent itemsets and generate association rules from transaction data for decision-making in retail and business analytics.

## Dataset
A sample retail transaction dataset containing **10 transactions** across **4 grocery items**: `Bread`, `Butter`, `Milk`, and `Eggs`.

| Transaction ID | Items Purchased |
| :---: | :--- |
| **T1** | Milk, Bread, Eggs |
| **T2** | Milk, Bread |
| **T3** | Milk, Eggs |
| **T4** | Bread, Butter |
| **T5** | Milk, Bread, Butter, Eggs |
| **T6** | Bread, Eggs |
| **T7** | Milk, Butter |
| **T8** | Milk, Bread, Butter |
| **T9** | Bread, Butter, Eggs |
| **T10** | Milk, Bread, Butter, Eggs |

*The transaction list is transformed into a One-Hot Encoded binary DataFrame prior to mining.*

## Algorithm / Steps Followed
1. Import required Python libraries (`pandas`, `mlxtend`).
2. Create the sample grocery transaction dataset.
3. Convert the dataset into a one-hot encoded binary matrix using `TransactionEncoder`.
4. Apply the `apriori()` algorithm (`min_support=0.3`) to identify frequent itemsets.
5. Generate association rules from frequent itemsets using `association_rules()` with `lift >= 1.0`.
6. Filter strong rules based on thresholds (`confidence >= 0.6` and `lift > 1`).
7. Interpret the rules to uncover cross-selling and store layout insights.

## Functions and Methods Used

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Data Handling** | `TransactionEncoder()` | Converts a list of transactions into a one-hot encoded binary matrix |
| **Frequent Pattern Mining** | `apriori()` | Finds frequent itemsets meeting the minimum support threshold |
| **Association Rule Generation** | `association_rules()` | Derives association rules using metrics such as support, confidence, and lift |
| **Filtering** | Conditional filtering (`rules[...]`) | Selects rules meeting strong confidence and lift thresholds |
| **Visualization / Display** | `print()` / `DataFrame.head()` | Displays encoded transactions, itemsets, and rules in tabular format |

## Program 
# Experiment 6: Market Basket Analysis (Association Rule Mining using Apriori Algorithm)

## Code

```python
# Market Basket Analysis using the Apriori Algorithm

# Step 1: Import required libraries
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

# Step 2: Create the sample grocery transaction dataset
transactions = [
    ["Milk", "Bread", "Eggs"],
    ["Milk", "Bread"],
    ["Milk", "Eggs"],
    ["Bread", "Butter"],
    ["Milk", "Bread", "Butter", "Eggs"],
    ["Bread", "Eggs"],
    ["Milk", "Butter"],
    ["Milk", "Bread", "Butter"],
    ["Bread", "Butter", "Eggs"],
    ["Milk", "Bread", "Butter", "Eggs"]
]

# Step 3: Convert to a one-hot encoded binary matrix
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
print("One-Hot Encoded Transactions:")
print(df)

# Step 4: Apply Apriori (min_support = 0.3)
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False).reset_index(drop=True)
print("\nFrequent Itemsets (min_support = 0.3):")
print(frequent_itemsets)

# Step 5: Generate association rules (lift >= 1.0)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]]
print("\nAssociation Rules (lift >= 1.0):")
print(rules.sort_values(by="lift", ascending=False).reset_index(drop=True))

# Step 6: Filter strong rules (confidence >= 0.6 and lift > 1)
strong_rules = rules[(rules["confidence"] >= 0.6) & (rules["lift"] > 1)]
strong_rules = strong_rules.sort_values(by=["lift", "confidence"], ascending=False).reset_index(drop=True)
print("\nStrong Association Rules (confidence >= 0.6 and lift > 1):")
print(strong_rules)
```
#OUTPUT
```
One-Hot Encoded Transactions:
   Bread  Butter   Eggs   Milk
0   True   False   True   True
1   True   False  False   True
2  False   False   True   True
3   True    True  False  False
4   True    True   True   True
5   True   False   True  False
6  False    True  False   True
7   True    True  False   True
8   True    True   True  False
9   True    True   True   True

Frequent Itemsets (min_support = 0.3):
    support                          itemsets
0       0.8                frozenset({Bread})
1       0.7                 frozenset({Milk})
2       0.6               frozenset({Butter})
3       0.6                 frozenset({Eggs})
4       0.5        frozenset({Butter, Bread})
5       0.5          frozenset({Eggs, Bread})
6       0.5          frozenset({Milk, Bread})
7       0.4         frozenset({Butter, Milk})
8       0.4           frozenset({Milk, Eggs})
9       0.3         frozenset({Butter, Eggs})
10      0.3  frozenset({Butter, Eggs, Bread})
11      0.3  frozenset({Butter, Milk, Bread})
12      0.3    frozenset({Milk, Eggs, Bread})

Association Rules (lift >= 1.0):
                   antecedents                 consequents  support  confidence      lift
0    frozenset({Butter, Eggs})          frozenset({Bread})      0.3    1.000000  1.250000
1           frozenset({Bread})   frozenset({Butter, Eggs})      0.3    0.375000  1.250000
2           frozenset({Bread})         frozenset({Butter})      0.5    0.625000  1.041667
3           frozenset({Eggs})          frozenset({Bread})      0.5    0.833333  1.041667
4           frozenset({Bread})           frozenset({Eggs})      0.5    0.625000  1.041667
5          frozenset({Butter})          frozenset({Bread})      0.5    0.833333  1.041667
6   frozenset({Butter, Bread})           frozenset({Eggs})      0.3    0.600000  1.000000
7     frozenset({Eggs, Bread})         frozenset({Butter})      0.3    0.600000  1.000000
8          frozenset({Butter})    frozenset({Eggs, Bread})      0.3    0.500000  1.000000
9          frozenset({Eggs})    frozenset({Butter, Bread})      0.3    0.500000  1.000000
10    frozenset({Milk, Bread})         frozenset({Butter})      0.3    0.600000  1.000000
11         frozenset({Butter})    frozenset({Milk, Bread})      0.3    0.500000  1.000000
12    frozenset({Milk, Bread})           frozenset({Eggs})      0.3    0.600000  1.000000
13          frozenset({Eggs})    frozenset({Milk, Bread})      0.3    0.500000  1.000000

Strong Association Rules (confidence >= 0.6 and lift > 1):
                 antecedents          consequents  support  confidence      lift
0  frozenset({Butter, Eggs})   frozenset({Bread})      0.3    1.000000  1.250000
1        frozenset({Butter})   frozenset({Bread})      0.5    0.833333  1.041667
2          frozenset({Eggs})   frozenset({Bread})      0.5    0.833333  1.041667
3         frozenset({Bread})  frozenset({Butter})      0.5    0.625000  1.041667
4         frozenset({Bread})    frozenset({Eggs})      0.5    0.625000  1.041667
```

## Result
Thus, Market Basket Analysis using the Apriori algorithm was executed successfully.

