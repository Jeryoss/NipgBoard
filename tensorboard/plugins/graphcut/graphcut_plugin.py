"""Plugin for Graphcut visualization.

This plugin provides a graphcut visualization tool for TensorBoard. It generates a graph of the
nearest neighbors and clusters them according to their proximity.

To use this plugin, run a TensorBoard server on the relevant log directory and go to the Graphcut
tab.

Example:
tensorboard --logdir /path/to/logs
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import json
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.algorithms import tree
from sklearn.metrics import confusion_matrix
from tensorboard import program
from tensorboard.backend.http_util import Respond
from tensorboard.compat import tf
from tensorboard.plugins import base_plugin
from werkzeug import wrappers

_PLUGIN_PREFIX_ROUTE = 'graphcut'

RUNS_ROUTE = '/runs'
PORT_ROUTE = '/port'
CLUSTER_ROUTE = '/clusters'


class GraphcutPlugin(base_plugin.TBPlugin):
    """Graphcut Plugin python script"""

    plugin_name = _PLUGIN_PREFIX_ROUTE

    def __init__(self, context):
        """
        Instantiates GraphcutPlugin.

        Args:
          context: A `base_plugin.TBContext` object.
        """
        import random as python_random
        import numpy as np

        # Set the seed value
        seed_value = 43

        # Set the random seed for Python
        python_random.seed(seed_value)

        # Set the random seed for NumPy
        np.random.seed(seed_value)


        self.multiplexer = context.multiplexer
        self.logdir = context.logdir
        self._handlers = None
        self.readers = {}
        self.run_paths = None
        self._configs = None
        self.old_num_run_paths = None
        self.config_fpaths = None
        self._is_active = False
        self._thread_for_determining_is_active = None
        if self.multiplexer:
            self.run_paths = self.multiplexer.RunPaths()

    def get_plugin_apps(self):
        """Returns a dict of routes offered by this plugin to the application.

        Returns:
          A dictionary where the keys are routes and the values are request handler functions.
        """
        self._handlers = {
            RUNS_ROUTE: self._serve_runs,
            PORT_ROUTE: self._serve_port,
            CLUSTER_ROUTE: self._serve_clusters,
        }
        return self._handlers

    def is_active(self):
        """Determines whether this plugin is active.

        Returns:
          A Boolean indicating whether this plugin is active.
        """
        return True

    @property
    def configs(self):
        """Returns the configs.

        Returns:
          The configs.
        """
        return self._configs

    def _run_paths_changed(self):
        """Determines whether the run paths have changed.

        Returns:
          A Boolean indicating whether the run paths have changed.
        """
        num_run_paths = len(list(self.run_paths.keys()))
        if num_run_paths != self.old_num_run_paths:
            self.old_num_run_paths = num_run_paths
            return True
        return False

    def _get_reader_for_run(self, run):
        """Returns a reader for a run.

        Args:
          run: The run for which to return a reader.

        Returns:
          A reader for the run.
        """
        if run in self.readers:
            return self.readers[run]

        config = self._configs[run]
        reader = None
        if config.model_checkpoint_path:
            try:
                reader = tf.pywrap_tensorflow.NewCheckpointReader(
                    config.model_checkpoint_path)
            except Exception:
                tf.logging.warning('Failed reading "%s"',
                                   config.model_checkpoint_path)
        self.readers[run] = reader
        return reader

    @wrappers.Request.application
    def _serve_runs(self, request):
        """Serve a list of available runs.

       Args:
           request (str): The request object.

       Returns:
           An HTTP response containing a list of available runs in JSON format.
        """
        keys = ['.']
        return Respond(request, list(keys), 'application/json')

    @wrappers.Request.application
    def _serve_port(self, request):
        """Serve the current port number.

        Args:
            request (str): The request object.

        Returns:
            An HTTP response containing the current port number in plain text format.
        """
        return Respond(request, str(program.global_port), "text/plain", 200)

    @wrappers.Request.application
    def _serve_clusters(self, request):
        """
        This method handles requests to cluster data using graph-based clustering.

        Args:
            request: The HTTP request object.

        Returns:
            A Respond object containing a JSON-formatted string with information about the clustering results.

        Raises:
            ValueError: If the input data is invalid or missing.

        """
        try:
            data = request.get_data().decode("utf-8")
            obj = json.loads(data)
            labels = np.array(obj["labels"], dtype=str)
            proj = np.array(obj["vectors"], dtype=np.float32)

            fraction_to_keep = float(request.args.get('fraction'))
            count = float(request.args.get('count'))
            sublogdir = request.args.get('sublogdir')
            dimension = request.args.get('dimension')

            # Open pre-saved TSNE data.
            proj = proj.reshape(len(labels), int(dimension))
            rows, cols = proj.shape

            # Generate graph.
            matrix = np.zeros(shape=(rows, rows))
            for i in range(rows):
                for j in range(rows):
                    matrix[i][j] = np.linalg.norm(proj[i][:cols] - proj[j][:cols])
            matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
            norm_matrix = 1 - matrix
            graph = nx.from_numpy_array(norm_matrix)

            # Get spanning tree.
            mst = tree.maximum_spanning_edges(graph, algorithm='kruskal')
            maximum_edgelist = list(mst)
            complete_edgelist = list(graph.edges(data=True))
            number_of_edges_to_keep = int(fraction_to_keep * len(complete_edgelist))
            sorted_weight_index_edgelist = sorted([[e[2]['weight'], i] for i, e in enumerate(complete_edgelist)],
                                                  reverse=True)
            str_edgelist = ['%s %s %s' % (str(edge[0]), str(edge[1]), str(edge[2])) for edge in maximum_edgelist]
            for i in range(len(proj), len(sorted_weight_index_edgelist)):
                if number_of_edges_to_keep <= 0:
                    break
                _, index = sorted_weight_index_edgelist[i]
                if complete_edgelist[index] not in maximum_edgelist:
                    str_edgelist.append('%s %s %s' % (
                        str(complete_edgelist[index][0]), str(complete_edgelist[index][1]),
                        str(complete_edgelist[index][2])))
                    number_of_edges_to_keep -= 1
            mst_graph = nx.parse_edgelist(str_edgelist)
            c = self.girvan_newman(mst_graph.copy(), count)
            node_groups = [list(i) for i in c]

            # Perform clustering and get results.
            points, predicted_labels, point_group, accuracy, accuracies = self.get_cluster_max_labels(node_groups,
                                                                                                      labels)
            classes, cm, tick_marks = self.plot_confusion_matrix(labels[points], predicted_labels, np.unique(labels),
                                                                 sublogdir)

            # Rename accuracies.
            accuracies = {f"{i}_{a.split('(')[-1].split(')')[0]}": v for i, (a, v) in enumerate(accuracies.items())}

            self.pairs_from_clusters(node_groups, len(labels), sublogdir)

            # Package results into a dictionary and return as JSON.
            results = {
                'classes': classes.tolist(),
                'cm': cm.tolist(),
                'tick_marks': tick_marks.tolist(),
                'nodes': predicted_labels,
                'accuracy': accuracy,
                'accuracies': accuracies,
                'groups': node_groups,
            }
        except (TypeError, TypeError, TypeError) as e:
            print(f"An error occurred while trying to read the configuration file: {e}")

        return Respond(request, json.dumps(results), 'text/plain', 200)

    def girvan_newman(self, graph: nx.Graph, count: int):
        """
        Divides the given graph into communities using Girvan-Newman algorithm.

        Args:
        - graph (nx.Graph): A NetworkX graph object representing the input graph.
        - count (int): The desired number of communities to divide the graph into.

        Returns:
        A list of sets, where each set represents a community of nodes in the graph.

        Raises:
        - ValueError: If the given count is greater than or equal to the number of nodes in the graph.
        """
        # find number of connected components
        np.random.seed(43)
        sg = nx.connected_components(graph)
        sg_count = nx.number_connected_components(graph)

        if count >= len(graph.nodes):
            raise ValueError("The desired count is greater than or equal to the number of nodes in the graph.")

        while (sg_count < count):
            graph.remove_edge(*self.edge_to_remove(graph))
            sg = nx.connected_components(graph)
            sg_count = nx.number_connected_components(graph)
        return list(sg)

    # def Louvain_community(self, graph: nx.Graph, count: int):
    #
    #     G = graph
    #     Louvain_resolution = 9  # @param {type:"slider", min:1, max:100, step:1}
    #     partition = community_louvain.best_partition(G, resolution=Louvain_resolution, randomize=True)
    #     # partition = community_louvain.best_partition(G, resolution=Louvain_resolution)
    #
    #     pos = nx.spring_layout(G)
    #     cmap = plt.cm.get_cmap('viridis', max(partition.values()) + 1)
    #     nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
    #                            cmap=cmap, node_color=list(partition.values()))
    #     nx.draw_networkx_edges(G, pos, alpha=0.9)
    #     plt.show()
    #     print("Number of clusters found: ", np.unique(np.array(list(partition.values()))).shape[0])
    #     print("Number of points clustred: ", np.unique(np.array(list(partition.keys()))).shape[0])
    #
    #
    #     Louvain_clusters = np.empty(np.unique(np.array(list(partition.values()))).shape, dtype=object)
    #     for key, value in partition.items():
    #         tmp_a = Louvain_clusters[value]
    #         if tmp_a is None:
    #             tmp_a = []
    #         tmp_a.append(key)
    #         Louvain_clusters[value] = tmp_a

    def edge_to_remove(self, graph: nx.Graph):
        """
        Finds the edge with the highest betweenness centrality score in the given graph.

        Args:
        - graph (nx.Graph): A NetworkX graph object representing the input graph.

        Returns:
        A tuple representing the edge with the highest betweenness centrality score.

        Raises:
        - ValueError: If the input graph is empty.
        """
        if len(graph.nodes) == 0:
            raise ValueError("Input graph is empty.")

        G_dict = nx.edge_betweenness_centrality(graph)
        edge = ()

        # extract the edge with highest edge betweenness centrality score
        for key, value in sorted(G_dict.items(), key=lambda item: item[1], reverse=True):
            edge = key
            break
        return edge

    def get_cluster_max_labels(self, node_groups, labels):
        """Returns the cluster information and the estimated accuracy of the clustering.

        Args:
            node_groups (list): A list of lists, where each sublist contains the indices of the nodes in a cluster.
            labels (array-like): An array of labels assigned to each node.

        Returns:
            tuple: A tuple containing:
                - points (list): A list of indices of the nodes.
                - predicted_lables (list): A list of predicted labels assigned to each node.
                - point_group (list): A list of indices indicating the corresponding cluster for each node.
                - accuracy (float): An estimated accuracy of the clustering.
                - accuracies (dict): A dictionary of the accuracy for each cluster.

        Raises:
            ValueError: If the length of `node_groups` does not match the number of unique labels in `labels`.
        """
        clusters_dict = {}
        clusters_count = 0

        # Count the labels in each cluster
        for i in node_groups:

            cluster = labels[np.array(i, dtype=np.int)]
            labels_dict = {}
            for j in cluster:
                if j not in labels_dict:
                    labels_dict[j] = 1
                else:
                    labels_dict[j] += 1

            total_cluster_count = 0
            max_value = 0
            max_percentage = 0
            max_label = ""

            for label in labels_dict:
                total_cluster_count += labels_dict[label]
                if labels_dict[label] > max_value:
                    max_value = labels_dict[label]
                    max_label = label
            max_percentage = max_value / total_cluster_count

            # the cluster belongs to this type
            clusters_dict[clusters_count] = {"label": max_label, "percentage": max_percentage, "count": max_value}
            clusters_count += 1
        print("Clusters information: ", clusters_dict)

        accuracies = {}
        TORF = []
        points = []
        predicted_lables = []
        point_group = []

        for i in range(len(node_groups)):
            print(clusters_dict)
            current_cluster_label = clusters_dict[i]['label']
            current_cluster_nums = []
            for j in node_groups[i]:
                point_group.append(i)
                points.append(int(j))
                predicted_lables.append(str(current_cluster_label))
                if labels[int(j)] == current_cluster_label:
                    TORF.append(1)
                    current_cluster_nums.append(1)
                else:
                    TORF.append(0)
                    current_cluster_nums.append(0)
            print(current_cluster_label)
            accuracies['Cluster' + str(i) + "(" + str(current_cluster_label) + ")"] = np.sum(
                current_cluster_nums) / len(current_cluster_nums)

        accuracy = np.sum(TORF) / len(TORF)
        print("Estimated accuracy: ", accuracy)
        return points, predicted_lables, point_group, accuracy, accuracies

    def pairs_from_clusters(self, node_groups, pairs_n, sublogdir):
        """
        Generates a binary matrix representing pairs of points that belong to the same cluster.

        Args:
        - node_groups (list of lists): List of cluster assignments, where each element of the outer list is a list of
                                       indices of points in the corresponding cluster.
        - pairs_n (int): Number of points in the dataset.
        - sublogdir (str): Subdirectory in the log directory where the output matrix will be saved.

        Returns: None
        """
        output_data = np.full((pairs_n, pairs_n), 0)
        for i in range(len(node_groups)):
            for j in range(len(node_groups[i])):
                for k in range(j, len(node_groups[i])):
                    output_data[j][k] = 1
        path = os.path.join(self.logdir, sublogdir, 'trainParams.npy')
        if os.path.exists(path):
            os.remove(path)
        np.save(path, output_data)

    def plot_confusion_matrix(self, y_test, y_pred, classes, sublogdir="", normalize=False, title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        Plot the confusion matrix for a classification model.

        Args:
            y_test (array-like): The true labels of the test set.
            y_pred (array-like): The predicted labels of the test set.
            classes (list): The list of class labels.
            sublogdir (str, optional): The subdirectory to save the plot in. Defaults to "".
            normalize (bool, optional): Whether to normalize the confusion matrix. Defaults to False.
            title (str, optional): The title of the plot. Defaults to 'Confusion matrix'.
            cmap (matplotlib colormap, optional): The color map to use for the plot. Defaults to plt.cm.Blues.

        Returns:
            tuple: A tuple containing the class labels, the confusion matrix, and the tick marks for the plot.

        """
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion matrix:\n", cm)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticklabels(classes)

        fmt = '.2f' if normalize else 'd'
        thresh = (cm.max() + cm.min()) / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color=color)

        fig.tight_layout()
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')

        if sublogdir:
            file = os.path.join(self.logdir, sublogdir, 'confusion.png')
        else:
            file = os.path.join(self.logdir, 'confusion.png')

        if os.path.isfile(file):
            os.remove(file)

        plt.savefig(file)

        return classes, cm, tick_marks
