"""Dataset group names and `--dataset_setting` resolution.

Both entry points used to carry their own copy of the group dict and of the
parse chain that turns `--dataset_setting` into dataset lists. The copies had
already drifted -- node classification renamed `link1`/`link2` to `div1`/`div2`
while keeping identical member lists -- so they live here instead, with the old
names kept as aliases.

A *side* of a `--dataset_setting` expression is a group name, or a
comma-separated list whose elements are dataset names, group names, or a mix.
`cora,citeseer+pubmed,products_home` trains on the first pair and tests on both
pairs; the group forms behave exactly as they did before.
"""

GROUPS = {
    'all': [
        'amazon-book', 'yelp2018', 'gowalla', 'yelp_textfeat', 'amazon_textfeat', 'steam_textfeat', 'Goodreads', 'Fitness', 'Photo', 'ml1m', 'ml10m', 'products_home', 'products_tech', 'cora', 'pubmed', 'citeseer', 'CS', 'arxiv', 'arxiv-ta', 'citation-2019', 'citation-classic', 'collab', 'ddi', 'ppa', 'proteins_spec0', 'proteins_spec1', 'proteins_spec2', 'proteins_spec3', 'email-Enron', 'web-Stanford', 'roadNet-PA', 'p2p-Gnutella06', 'soc-Epinions1'
    ],
    'ecommerce': [
        'amazon-book', 'yelp2018', 'gowalla', 'yelp_textfeat', 'amazon_textfeat', 'steam_textfeat', 'Goodreads', 'Fitness', 'Photo', 'ml1m', 'ml10m', 'products_home', 'products_tech'
    ],
    'academic': [
        'cora', 'pubmed', 'citeseer', 'CS', 'arxiv', 'arxiv-ta', 'citation-2019', 'citation-classic', 'collab'
    ],
    'others': [
        'ddi', 'ppa', 'proteins_spec0', 'proteins_spec1', 'proteins_spec2', 'proteins_spec3', 'email-Enron', 'web-Stanford', 'roadNet-PA', 'p2p-Gnutella06', 'soc-Epinions1'
    ],
    'link1': [
        'products_tech', 'yelp2018', 'yelp_textfeat', 'products_home', 'steam_textfeat', 'amazon_textfeat', 'amazon-book', 'citation-2019', 'citation-classic', 'pubmed', 'citeseer', 'ppa', 'p2p-Gnutella06', 'soc-Epinions1', 'email-Enron',
    ],
    'link2': [
        'Photo', 'Goodreads', 'Fitness', 'ml1m', 'ml10m', 'gowalla', 'arxiv', 'arxiv-ta', 'cora', 'CS', 'collab', 'proteins_spec0', 'proteins_spec1', 'proteins_spec2', 'proteins_spec3', 'ddi', 'web-Stanford', 'roadNet-PA',
    ],
}

# Node classification reads everything from the node_data/ subdirectory, which
# holds only the five datasets in 'node'. The link-prediction groups below are
# inherited and remain resolvable for compatibility, but naming one will fail on
# a missing directory -- they were never usable here.
NODE_GROUPS = dict((name, list(members)) for name, members in GROUPS.items())
NODE_GROUPS['div1'] = list(GROUPS['link1'])  # legacy alias
NODE_GROUPS['div2'] = list(GROUPS['link2'])  # legacy alias
NODE_GROUPS['node'] = ['cora', 'arxiv', 'pubmed', 'home', 'tech']


def known_datasets(groups):
    """Every dataset name reachable through `groups`.

    Deliberately the union of all the values, not `groups['all']`: `home` and
    `tech` belong to NODE_GROUPS['node'] but not to its 'all'.
    """
    return set(name for members in groups.values() for name in members)


def resolve_side(spec, groups, known, context=None):
    """Resolve one side of a `--dataset_setting` expression to dataset names.

    `spec` is a group name, or a comma-separated list of dataset and/or group
    names. Groups are flattened; the result is de-duplicated, order-preserving.
    `context` is the full setting, quoted back in errors so a bad token in one
    side of an `A+B` names what the user actually typed.
    """
    context = spec if context is None else context
    if spec in groups:
        return list(groups[spec])
    out, seen = [], set()
    for part in (p.strip() for p in spec.split(',')):
        if not part:
            continue
        members = groups[part] if part in groups else [part]
        for name in members:
            if name not in known:
                raise SystemExit(
                    "unknown dataset or group '%s' in --dataset_setting '%s'. "
                    "Groups: %s." % (name, context, ', '.join(sorted(groups)))
                )
            if name not in seen:
                seen.add(name)
                out.append(name)
    if not out:
        raise SystemExit("--dataset_setting '%s' resolved to no datasets" % context)
    return out


def resolve_dataset_setting(setting, groups):
    """Turn `--dataset_setting` into (trn_datasets, tst_datasets_group).

    The result splats straight into MultiDataHandler. Under `A+B` the model
    trains on A and is tested against two groups, A (full-shot) and B
    (zero-shot); every other form tests on a single group.

    `_in_` is matched before `+` because it is the longer, more specific token.
    No dataset or group name contains either, so this cannot change how an
    existing spec resolves.
    """
    known = known_datasets(groups)
    if '_in_' in setting:
        left, right = setting.split('_in_', 1)
        left = resolve_side(left, groups, known, setting)
        right = resolve_side(right, groups, known, setting)
        both = [name for name in left if name in right]
        if not both:
            raise SystemExit(
                "--dataset_setting '%s' resolved to no datasets: the two sides "
                "do not overlap" % setting
            )
        return both, [both]
    if '+' in setting:
        left, right = setting.split('+', 1)
        trn = resolve_side(left, groups, known, setting)
        tst = resolve_side(right, groups, known, setting)
        return trn, [trn, tst]
    both = resolve_side(setting, groups, known, setting)
    return both, [both]
