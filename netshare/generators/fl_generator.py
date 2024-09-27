from netshare.generators.generator import Generator
import copy
from netshare import model_managers

class FederatedGenerator(Generator):
    def __init__(self, config, clients):
        super(FederatedGenerator, self).__init__(config)
        self.clients = clients  # List of client identifiers or data paths
        self.global_generator = None  # Placeholder for global generator model
        self.client_generators = []  # List to hold each client's generator model

    def initialize_clients(self):
        for client_id in self.clients:
            # Initialize each client's generator and discriminator models with specified model manager
            client_model_manager = model_managers.ModelManager(  # Replace with the correct class if available in model_managers
                config=self._config['model_manager']['config']
            )
            self.client_generators.append(client_model_manager)

    def train_local(self, client_index, epochs=1):
        # Train the local generator model for a specific client
        client_generator = self.client_generators[client_index]
        preprocessed_data = self._pre_post_processor.preprocess(self._ori_data_path)
        
        for epoch in range(epochs):
            client_generator.train(preprocessed_data)  # Train using local data
            
            # Optional: add differential privacy mechanism here
            # clipped_grads = self.clip_gradients(client_generator)
            # noisy_grads = self.add_noise(clipped_grads)

            # Print message indicating progress of specific client and epoch
            client_id = self.clients[client_index]
            print(f"Client {client_id} finished epoch {epoch + 1}.")

    def aggregate_global_model(self):
        # Aggregate models using Federated Averaging (FedAvg)
        total_clients = len(self.client_generators)
        new_global_params = copy.deepcopy(self.client_generators[0].model.get_weights())
        
        for param_index in range(len(new_global_params)):
            for client_generator in self.client_generators[1:]:
                client_params = client_generator.model.get_weights()
                new_global_params[param_index] += client_params[param_index]
            
            new_global_params[param_index] /= total_clients

        # Update the global generator model with the new aggregated parameters
        if self.global_generator is None:
            self.global_generator = copy.deepcopy(self.client_generators[0].model)
        
        self.global_generator.set_weights(new_global_params)
        print("Global model updated with Federated Averaging.")

    def distribute_global_model(self):
        # Send the global model parameters to each client's local model
        global_params = self.global_generator.get_weights()
        for client_generator in self.client_generators:
            client_generator.model.set_weights(global_params)
        print("Global model parameters distributed to all clients.")

    def train_federated(self, rounds=1, local_epochs=1):
        # Main function to train the federated model
        self.initialize_clients()
        
        for round_index in range(rounds):
            print(f"Starting federated round {round_index + 1}.")
            # Each client trains their local model
            for client_index in range(len(self.clients)):
                self.train_local(client_index, epochs=local_epochs)

            # Aggregate client models to update the global model
            self.aggregate_global_model()

            # Distribute the updated global model to all clients
            self.distribute_global_model()
            
        print("Federated training completed.")