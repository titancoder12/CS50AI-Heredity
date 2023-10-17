import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)
    
    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    #print(f"\n*************\none_gene:{one_gene}, \ntwo_gene:{two_genes}, \nhave_trait: {have_trait}\n")
    probabilities = []
    no_genes = []
    for person in people.keys():
        if (person not in one_gene) and (person not in two_genes):
            no_genes.append(person)

    for person in people.keys():
        mother = people[person]["mother"]
        father = people[person]["father"]
        motherprob = 0
        fatherprob = 0
        
        if mother in no_genes:
            motherprob = PROBS["mutation"]
        if mother in one_gene:
            motherprob = 0.5 - PROBS["mutation"]
        if mother in two_genes:
            motherprob = 1 - PROBS["mutation"]
        
        if father in no_genes:
            fatherprob = PROBS["mutation"]
        if father in one_gene:
            fatherprob = 0.5 - PROBS["mutation"]
        if father in two_genes:
            fatherprob = 1 - PROBS["mutation"]

        if (mother == None) and (father == None):
            if (person not in one_gene) and (person not in two_genes):
                probabilities.append(PROBS["gene"][0])
            if person in one_gene:
                probabilities.append(PROBS["gene"][1])
            if person in two_genes:
                probabilities.append(PROBS["gene"][2])
        else:
            if person in no_genes:
                print(f"{person} in no_gene")
                probabilities.append((1 - motherprob) * (1 - fatherprob))

            if person in one_gene:
                print(f"{person} in one_gene")
                probabilities.append(((1 - motherprob) * (fatherprob)) + ((1-fatherprob) * (motherprob)))
            
            if person in two_genes:
                print(f"{person} in two_gene")
                probabilities.append((motherprob) * (fatherprob))
        
        # PROBS for no trait
        if person not in have_trait:
            if person in no_genes:
                probabilities.append(PROBS["trait"][0][False])
            if person in one_gene:
                probabilities.append(PROBS["trait"][1][False])
            if person in two_genes:
                probabilities.append(PROBS["trait"][2][False])
        
        # PROBS for trait
        if person in have_trait:
            if person in no_genes:
                probabilities.append(PROBS["trait"][0][True])
            if person in one_gene:
                probabilities.append(PROBS["trait"][1][True])
            if person in two_genes:
                probabilities.append(PROBS["trait"][2][True])
        
    probabilities_multiplied = 1
    for probability in probabilities:
        probabilities_multiplied = probabilities_multiplied * probability
    
    #print(f"Joint Probability: {probabilities_multiplied}")
    return probabilities_multiplied


    #raise NotImplementedError


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        # Probabilities for genes
        if (person not in one_gene) and (person not in two_genes):
            probabilities[person]["gene"][0] += p
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        if person in two_genes:
            probabilities[person]["gene"][2] += p
        
        # Probabilities for traits
        if person in have_trait:
            probabilities[person]["trait"][True] += p
        if person not in have_trait:
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for name in probabilities:
        person = probabilities[name].copy()
        #name = list(person.keys())[0]

        total_genes = sum([person["gene"][0], person["gene"][1], person["gene"][2]])

        # Normalize genes
        probabilities[name]["gene"][0] = person["gene"][0] / total_genes
        probabilities[name]["gene"][1] = person["gene"][1] / total_genes
        probabilities[name]["gene"][2] = person["gene"][2] / total_genes
        
        #print(probabilities[name]["gene"])

        # Normalize traits
        total_traits = sum([person["trait"][True], person["trait"][False]])

        probabilities[name]["trait"][True] = person["trait"][True] / total_traits
        probabilities[name]["trait"][False] = person["trait"][False] / total_traits

        #print(probabilities)
        #print("total traits:")
        #print(total_traits)

    for person in probabilities:
        p = probabilities[person]
        #print("gene:")
        #print(sum([p["gene"][0], p["gene"][1], p["gene"][2]]))

        #print("trait:")
        #print(sum([p["trait"][True], p["trait"][False]]))



if __name__ == "__main__":
    main()
