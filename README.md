# Mattermost-Recommendations

CERN Mattermost Dataset Recommender System Evaluation

## Abstract

Recommender systems play a pivotal role in various human-centered online systems by filtering out relevant information from a large information base. However, most recommender systems consume explicit private user information and information between users and items without exploring other latent factors. In this work, we build and analyze implicit social networks to discover and extract measures indicating the strengths and similarities of relationships between users and channels. These measures are used for collaborative filter-based recommender systems, where their effects and performance are compared and evaluated against simple measures.

## Introduction ☕️

### Dataset Description

Mattermost is an open-source communication platform similar to slack that is widely used at CERN. The CERN Anonymized Mattermost Dataset includes Mattermost data from January 2018 to November 2021 with 20794 CERN  users, 2367 Mattermost teams, 12773 Mattermost channels, 151 CERN buildings, and 163 CERN  organizational units. The data set states the relationship between Mattermost teams, Mattermost channels, and CERN users, and holds various information such as channel creation, channel deletion times, user channel joining and leave times, and user-specific information such as building and organizational units. To hide identifiable information (e.g. Team Name, User Name, Channel Name, etc.), the dataset was anonymized. The anonymization was done by omitting some attributes, hashing string values, and removing connections between users/teams/channels.

Dataset License: ***CC BY-NC Creative Commons Attribution Non-Commercial Licence***

Dataset Link: CERN Anonymized Mattermost Data | [Zenodo](https://zenodo.org/record/6319684#.YnOMdi8Rr0o)

```bibtex
@dataset{jakovljevic_igor_2022_6319684,
  author       = {Jakovljevic, Igor and
                  Wagner, Andreas and
                  Gütl, Christian and
                  Pobaschnig, Martin and
                  Mönnich, Adrian},
  title        = {CERN Anonymized Mattermost Data},
  month        = mar,
  year         = 2022,
  publisher    = {Zenodo},
  version      = 1,
  doi          = {10.5281/zenodo.6319684},
  url          = {https://doi.org/10.5281/zenodo.6319684}
}

```

---

## Getting Started 🏁

1. Initialize the repository and download data

```sh
make all
```

2. The project consists of two seperate component. The data analysis and the recommender system compoenent.
    1. Install necessary requirements for recommender system:

      ```sh
      make recommender
      ```

    2. Install necessary requirements for data analysis:

      ```sh
      make analysis
      ```

3. Run Jupyter Lab:

```sh
make start
```

---

## Involved institutions 🏫

Contributors from the following institutions were involved in the development of this project:

* [CERN](https://home.cern/)
* [Graz University of Technology](https://www.tugraz.at/home/)

---

## Acknowledgements 🙏

We would like to express our gratitude to CERN, for allowing us to publish the dataset as open data and use it for research purposes.
