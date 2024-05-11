import torch
import numpy as np
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
from scipy.stats import pearsonr, spearmanr
import os
import pickle
import wandb
from tqdm import tqdm
from topoformer.models.constraints import Constraints

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttentionVisualizer:
    """
    A class to visualize the attention mechanism in a given model.
    """
    def __init__(
        self,
        model,
        layer_to_probe,
        test_dataloader,
        pooling="AVERAGE",
        n_examples=500,
        map_side=20,
        save_images=False,
        modality="images",
        images_dir=None,
    ):
        self.model = model
        self.test_dataloader = test_dataloader
        self.attention_dict = {}
        self.layer_to_probe = layer_to_probe
        self.n_examples = n_examples
        self.map_side = map_side
        self.save_images = save_images
        self.modality = modality
        self.images_dir = images_dir
        if self.save_images:
            assert self.images_dir is not None, "Please provide an images directory"
            os.makedirs(self.images_dir, exist_ok=True)
        
        if self.modality == 'text':
            self.positive_examples = []
            self.negative_examples = []

        # convert test_dataset so that it has batch_size of 1 and only n_examples
        if self.modality == 'images':
            self.test_dataloader = self._create_visualization_dataloader(
                test_dataloader=self.test_dataloader, num_samples=self.n_examples
            )

    def separate_examples(self):
        for i in self.test_dataloader:
            if i['label'] == 0:
                self.positive_examples.append(i)
            else:
                self.negative_examples.append(i)
        self.positive_examples = self.positive_examples[:self.n_examples]
        self.negative_examples = self.negative_examples[:self.n_examples]

    def hook_fn(self, module, input, output):
        self.attention_dict["attention"].append(output.mean(dim=1).detach().cpu())

    def _create_visualization_dataloader(
        self, test_dataloader, num_samples, batch_size=1
    ):

        original_dataset = test_dataloader.dataset

        small_dataset = Subset(original_dataset, range(num_samples))

        small_dataloader = DataLoader(
            small_dataset, batch_size=batch_size, shuffle=False
        )

        return small_dataloader

    def _register_hook(self, layer_name):
        self.attention_dict["attention"] = []
        hook = None
        for name, layer in self.model.named_modules():
            if name == layer_name:
                hook = layer.register_forward_hook(self.hook_fn)
        if hook is None:
            raise ValueError(f"Layer {layer_name} not found!")
        return hook

    def process_examples(self, examples):
        hook = self._register_hook(self.layer_to_probe)
        for example in examples:
            if self.modality == 'images':
                batch = [t.to(DEVICE) for t in example]
                images, labels = batch
                returns = self.model(images)  # noqa: F841
            else:
                ids = example["ids"].to(DEVICE)
                returns = self.model(ids)  # noqa: F841
        hook.remove()

        attention_data = torch.stack(list(self.attention_dict.values())[0]).squeeze(1)
        print(
            f" This is the attention shape, should be {self.n_examples}, {self.map_side**2}: {attention_data.shape}"
        )
        # If modality is text, then reshape to plot selectivities later.
        if self.modality != 'images':
            attention_data = attention_data.reshape(
            self.n_examples, self.map_side, self.map_side
        )
        return attention_data

    def _visualize_images(self) -> None:
        X = self.process_examples(self.test_dataloader)
        print(f"shape is {X.shape}")
        pca, components, variance = self.apply_pca(X)
        self.plot_pca_results(pca, components, variance)

    def _visualize_text(self) -> None:
        self.separate_examples()

        x_positive = self.process_examples(self.positive_examples)
        x_negative = self.process_examples(self.negative_examples)
        self.plot_mean_activations(x_positive, x_negative)
        p_values, _ = self.calculate_p_values(x_positive, x_negative)

        x_positive_flat = x_positive.reshape(x_positive.shape[0], -1)
        x_negative_flat = x_negative.reshape(x_negative.shape[0], -1)
        concatenated_data = np.concatenate((x_positive_flat, x_negative_flat))
        assert x_positive_flat.shape == (self.n_examples, self.map_side**2)
        assert x_negative_flat.shape == (self.n_examples, self.map_side**2)
        print(f"shape is {x_positive_flat.shape}")
        print(f"shape is {x_positive_flat.shape}")
        self.x_positive_flat = x_positive_flat
        self.x_negative_flat = x_negative_flat

        pca, components, variance = self.apply_pca(concatenated_data)
        self.plot_p_values_heatmap(p_values)
        self.plot_pca_results(pca, components, variance)
        self.calculate_tgi(concatenated_data)

    def visualize_attention(self) -> None:
        visualization_strategy = {
            'text': self._visualize_text,
            'images': self._visualize_images,
        }
        try:
            visualization_strategy[self.modality]()
        except KeyError:
            raise ValueError(f"Modality {self.modality} is not supported.")


    def calculate_p_values(self, positive, negative):
        p_values = []
        t_values = []
        num_rows, num_columns = positive.shape[1], positive.shape[2]
        for row in range(num_rows):
            for column in range(num_columns):
                t_value, p_value = ttest_ind(
                    positive[:, row, column], negative[:, row, column]
                )
                p_values.append(p_value)
                t_values.append(t_value)

        p_values = np.array(p_values)
        p_values = -np.log10(p_values) * np.sign(np.array(t_values))
        p_values = p_values.reshape(self.map_side, self.map_side)
        return p_values, t_values
    
    def plot_p_values_heatmap(self, p_values):
        title = log_title = f"Heatmap of signed log(p-values)"
        self.plot_heatmap(p_values, title, log_title)

    def apply_pca(self, attention_data, n_components=10, explained_variance_cutoff=5):
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(attention_data)
        print(f"transformed data shape inside apply pca: {transformed_data.shape}")
        components = pca.components_

        pca_variance = PCA(n_components=explained_variance_cutoff)
        pca_variance.fit(attention_data)
        # transformed data: PCA dimensionality reduces
        self.transformed_data = transformed_data
        # Weights vectors(Eigen vectors)
        self.components = components
        self.variance_explained = pca_variance.explained_variance_ratio_
        return transformed_data, components, pca_variance.explained_variance_ratio_

    def plot_heatmap(self, data, title, log_title, save_name=None):
        plt.figure()
        sns.heatmap(data, center=0, square=True)
        plt.title(title)
        wandb.log({f"{self.layer_to_probe}_{log_title}": wandb.Image(plt)})
        if self.save_images:
            plt.savefig(
                f"{self.images_dir}/{save_name}_{self.layer_to_probe}.png"
            )
            plt.close()
        else:
            plt.show()

    def plot_pca_results(self, data, components, variance):
        titles = [f"PC{i+1} weights" for i in range(components.shape[0])]

        # Plot the components first two only:
        for idx, component in enumerate(components[:2]):
            self.plot_heatmap(component.reshape(self.map_side, self.map_side), 
                              f"{titles[idx]} for layer: {self.layer_to_probe}", 
                              f"PCA weights: {titles[idx]}", 
                              f"pca_weights_{idx}")

        # Plot the variance
        plt.plot(range(0, len(variance)), variance)
        plt.ylabel("Explained Variance")
        plt.xlabel("Principal Components")
        plt.xticks(
            range(0, len(variance)),
            [f"Component: {i+1}" for i in range(len(variance))],
            rotation=60,
        )
        plt.title("Explained Variance Ratio")
        if self.save_images:
            plt.savefig(f"{self.images_dir}/pca_variance{self.layer_to_probe}.png")
            plt.close()
        else:
            plt.show()

    def plot_mean_activations(self, positive, negative):
        self.plot_heatmap(positive.mean(axis=0), 
                          "Positive Mean Activations", 
                          "positive_mean_activations", 
                          "positive_mean_activations")
        self.plot_heatmap(negative.mean(axis=0), 
                          "Negative Mean Activations", 
                          "negative_mean_activations", 
                          "negative_mean_activations")


    @staticmethod
    def compute_generic_summary_stat(activations, distances, n_samples=100000, plot=False, stat_type='rankcorr', max_dist=None):
        """
        computes the inverse-distance-scaled sum of pairwise response correlations

        Args:
            activations: NxP matrix
            distances: PxP matrix
            n_samples: number of random pairs of units to sample
            plot: whether to produce a plot
            stat_type: how to aggregate the distance and correlation values. options: ['standard', 'corr', 'rankcorr']
            max_dist: maximum distance over which to calculate the statistic
        Returns:
            stat: inverse-distance-scaled sum of pairwise response correlations
        """
        
        # get rid of nans and infinities
        activations[np.isnan(activations)] = 0
        activations[activations == np.inf] = np.nanmax(activations)

        # compute pairwise correlations
        corrs = np.corrcoef(activations.transpose())
        corrs = corrs.reshape(-1)
        corrs[np.isnan(corrs)] = 0
        distances = distances.reshape(-1)

        ms = np.sqrt(distances.shape[0]) # map side

        if max_dist is not None:
            # select only units within max_dist
            inds = distances <= max_dist
            distances = distances[inds]
            corrs = corrs[inds]

        # ensure we are not looking at same-unit pairs
        corrs = corrs[distances != 0]
        distances = distances[distances != 0]

        # subselect for efficiency
        rand_inds = np.random.choice(corrs.shape[0], size=np.minimum(n_samples, corrs.shape[0]), replace=False)
        dists = distances[rand_inds]
        corrs = corrs[rand_inds]

        # standardize correlations so that networks with totally correlated units (and therefore a useless representational space) don't dominate
        corrs = (corrs - np.mean(corrs))/np.std(corrs)
        # finally compute summary stat
        stat_calculators = {
            'standard': lambda corrs, dists: np.mean(corrs/dists),
            'corr': lambda corrs, dists: -pearsonr(corrs, dists)[0],
            'rankcorr': lambda corrs, dists: -spearmanr(corrs, dists)[0]
        }
        try:
            stat = stat_calculators[stat_type](corrs, dists)
        except KeyError:
            raise NotImplementedError(f"Stat type {stat_type} is not implemented.")

        if plot:
            fig, ax = plt.subplots(figsize=(3,3))
            ax.plot(ms*dists, corrs, 'o', alpha=0.01)
            ax.set_xlim(0, ms*1.01*np.max(dists))
            ax.axhline(0.0, 0, 1, color='r', linestyle='--')
            ax.set_ylim(-1.05, 1.05)
            ax.set_xlabel('Pairwise unit distance (unit lengths)')
            ax.set_ylabel('Pairwise unit response correlation')
            ax.set_title(f'stat={stat:.04f}')
            # im_dom = domain if domain is not None else 'all'
            plt.show()

        return stat 
    
    def calculate_tgi(self, concatenated_data, max_dist=None, **kwargs):
        distances = Constraints(self.map_side**2, 0.3, True).distances # We're not doing anything with rwid here so it's fine
        max_dists = np.linspace(0.1, 0.8, 8)
        stats = []
        
        # Calculate TGI for each max_dist
        for distance in max_dists:
            result = self.compute_generic_summary_stat(concatenated_data, distances, max_dist=distance, **kwargs)
            stats.append(result)

        stats_array = np.array(stats)
        # Take average of stats
        stat = np.mean(stats_array)

        tag = f'_maxdist-{max_dist}' if max_dist is not None else ''
        wandb.log({f"tgi{tag}_layer_{self.layer_to_probe}": stat})


class AttentionVisualizerV2:
    def __init__(
        self,
        model,
        layers_to_probe,
        test_dataloader,
        map_side=20,
        save_images=False,
        no_wandb=False,
        prepend_name='',
        images_dir=None,
        token_level=False,
    ):
        self.model = model
        self.device = DEVICE
        self.test_dataloader = test_dataloader
        self.attention_dict = {}
        self.layers_to_probe = layers_to_probe
        self.map_side = map_side
        self.save_images = save_images
        self.images_dir = images_dir
        if self.save_images:
            assert self.images_dir is not None, "Please provide an images directory"
            os.makedirs(self.images_dir, exist_ok=True)
        self.no_wandb = no_wandb
        self.prepend_name = prepend_name
        self.layers = [layer.replace(prepend_name, '') for layer in layers_to_probe]
        self.token_level = token_level

    def hook_fn(self, module, input, output):
        # Take mean of tokens, should make this a parameter to pass!
        # get name of module
        for name, layer in self.model.named_modules():
            if layer == module:
                layer_name = name.replace(self.prepend_name, '')
                break
        if self.token_level:
            self.attention_dict[layer_name].append(output.detach().cpu())
        else:
            self.attention_dict[layer_name].append(output.mean(dim=1).detach().cpu())

    def _create_visualization_dataloader(
        self, test_dataloader, num_samples, batch_size=1
    ):

        original_dataset = test_dataloader.dataset

        small_dataset = Subset(original_dataset, range(num_samples))

        small_dataloader = DataLoader(
            small_dataset, batch_size=batch_size, shuffle=False
        )

        return small_dataloader

    def _register_hook(self, layer_name):
        self.attention_dict[layer_name.replace(self.prepend_name, '')] = []
        hook = None
        for name, layer in self.model.named_modules():
            if name == layer_name:
                hook = layer.register_forward_hook(self.hook_fn)
        if hook is None:
            raise ValueError(f"Layer {layer_name} not found!")
        return hook

    def process_examples(self, examples):
        hooks = {}
        all_labels = []
        for layer in self.layers_to_probe:
            hooks[layer] = self._register_hook(layer)
        for batch in tqdm(examples):
            batch = [t.to(DEVICE) for t in batch]
            images, labels = batch
            returns = self.model(images)  # noqa: F841
            all_labels += labels
        for hook in hooks.values():
            hook.remove()

        self.labels = torch.stack(all_labels).detach().cpu()

        for layer in self.layers:
            self.attention_dict[layer] = torch.concatenate(self.attention_dict[layer], 0).squeeze()

        # attention_data = torch.concatenate(self.attention_dict['attention'], 0).squeeze()
        # attention_data = torch.stack(list(self.attention_dict.values())[0]).squeeze(1)
        # print(
        #     f" This is the attention shape, should be {self.n_examples}, {self.map_side**2}: {attention_data.shape}"
        # )
        return self.attention_dict

    def visualize_attention(self) -> None:

        with torch.no_grad():
            X_all = self.process_examples(self.test_dataloader)

        for layer in self.layers:
            print(layer)
            X = X_all[layer]

            if self.token_level:
                X = X.reshape(-1, X.shape[-1])

            pca, components, variance = self.apply_pca(X)
            self.plot_pca_results(pca, components, variance, layer)

    def apply_pca(self, attention_data, n_components=10, explained_variance_cutoff=5):
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(attention_data)
        # print(f"transformed data shape inside apply pca: {transformed_data.shape}")
        components = pca.components_

        pca_variance = PCA(n_components=explained_variance_cutoff)
        pca_variance.fit(attention_data)
        # transformed data: PCA dimensionality reduces
        self.transformed_data = transformed_data
        # Weights vectors(Eigen vectors)
        self.components = components
        self.variance_explained = pca_variance.explained_variance_ratio_
        return transformed_data, components, pca_variance.explained_variance_ratio_

    def plot_pca_results(self, data, components, variance, layer):
        titles = [f"PC{i+1} weights" for i in range(components.shape[0])]

        # Plot the components first two only:
        for idx, component in enumerate(components[:2]):
            plt.figure()
            sns.heatmap(component.reshape(self.map_side, self.map_side), center=0, robust=True)
            plt.title(f"{titles[idx]} for layer: {layer}")
            if not self.no_wandb:
                wandb.log(
                    {f"{layer} PCA weights: {titles[idx]}": wandb.Image(plt)}
                )
            if self.save_images:
                plt.savefig(
                    f"{self.images_dir}/pca_weights_{idx}_{layer}.pdf"
                )
                plt.close()
            else:
                plt.show()

        # Plot the variance
        plt.plot(range(0, len(variance)), variance)
        plt.ylabel("Explained Variance")
        plt.xlabel("Principal Components")
        plt.xticks(
            range(0, len(variance)),
            [f"Component: {i+1}" for i in range(len(variance))],
            rotation=60,
        )
        plt.title("Explained Variance Ratio")
        if self.save_images:
            plt.savefig(f"{self.images_dir}/pca_variance{layer}.pdf")
            plt.close()
        else:
            plt.show()
