import time
import json
import hashlib
from uuid import uuid4

from flask import Flask, request, jsonify
from urllib.parse import urlparse
import requests


class BlockChain:
    """
    Block chain class to create the new blocks, transactions and hold the master chain.
    """

    def __init__(self):
        """
        Constructor to create a new block chain with the genesis block and empty set of nodes.
        """
        self.chain = []
        self.current_transactions = []
        self.new_block(previous_hash=1, proof=100)
        self.nodes = set()

    def new_block(self, proof, previous_hash=None):
        """
        Function to add a new block to the existing chain.
        :param proof: The proof given by the proof of work algorithm.
        :param previous_hash: Previous hash of the block
        :return: New block
        """
        block = {'index': len(self.chain) + 1, 'timestamp': time.time(), 'transactions': self.current_transactions,
                 'proof': proof, 'previous_hash': previous_hash or self.hash(self.chain[-1])}

        # Now reset the current list of transactions
        self.current_transactions = []
        self.chain.append(block)
        return block

    def new_transaction(self, sender, receiver, amount):
        """
        Add a new transaction to the chain.
        :param sender: The sender's address
        :param receiver: The receiver's address
        :param amount: Amount to be transferred.
        :return: The index of the block that will hold the transaction.
        """
        self.current_transactions.append({'sender': sender, 'receiver': receiver, 'amount': amount})
        return self.last_block['index'] + 1

    @staticmethod
    def hash(block):
        """
        Creates a SHA-512 hash of a Block
        :param block:
        :return:
        """
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha512(block_string).hexdigest()

    @property
    def last_block(self):
        """
        Returns the current last block of the chain.
        :return:
        """
        return self.chain[-1]

    def proof_of_work(self, last_proof):
        """
        Simple proof of work algorithm.
            - Find a number p' such that hash(pp') contains leading 4 zeroes, where p is the previous p'
            - p is the previous proof, and p' is the new proof
        :param last_proof:
        :return:
        """
        proof = 0
        while not self.valid_proof(last_proof, proof):
            proof += 1

        return proof

    @staticmethod
    def valid_proof(last_proof, current_proof):
        """
        Validates the Proof: Does hash(last_proof, proof) contain 4 leading zeroes?
        :param last_proof:
        :param current_proof:
        :return:
        """
        guess = '{0}{1}'.format(last_proof, current_proof).encode()
        guess_hash = hashlib.sha512(guess).hexdigest()
        return guess_hash[:4] == "0000"

    def register_nodes(self, address):
        """
        Add a new node to the list of nodes
        :param address: Address of node. Eg. 'http://192.168.0.5:5000'
        :return: None
        """
        parsed_url = urlparse(address)
        self.nodes.add(parsed_url.netloc)

    def valid_chain(self, chain):
        """
        Determine if a given block_chain is valid
        :param chain: <list> A block_chain
        :return: <bool> True if valid, False if not
        """

        last_block = chain[0]
        current_index = 1

        while current_index < len(chain):
            block = chain[current_index]
            print(last_block)
            print(block)
            print("\n-----------\n")
            # Check that the hash of the block is correct
            if block['previous_hash'] != self.hash(last_block):
                return False

            # Check that the Proof of Work is correct
            if not self.valid_proof(last_block['proof'], block['proof']):
                return False

            last_block = block
            current_index += 1

        return True

    def resolve_conflicts(self):
        """
        This is our Consensus Algorithm, it resolves conflicts
        by replacing our chain with the longest one in the network.
        :return: <bool> True if our chain was replaced, False if not
        """

        neighbours = self.nodes
        new_chain = None

        # We're only looking for chains longer than ours
        max_length = len(self.chain)

        # Grab and verify the chains from all the nodes in our network
        for node in neighbours:
            response = requests.get('http://{0}/chain'.format(node))

            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']

                # Check if the length is longer and the chain is valid
                if length > max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain

        # Replace our chain if we discovered a new, valid chain longer than ours
        if new_chain:
            self.chain = new_chain
            return True

        return False


# Instantiate our Node
app = Flask(__name__)

# Generate a globally unique address for this node
node_identifier = str(uuid4()).replace('-', '')

# Instantiate the BlockChain
block_chain = BlockChain()


@app.route('/mine', methods=['GET'])
def mine():
    # We run the proof of work algorithm to get the next proof...
    last_block = block_chain.last_block
    last_proof = last_block['proof']
    proof = block_chain.proof_of_work(last_proof)

    # We must receive a reward for finding the proof.
    # The sender is "0" to signify that this node has mined a new coin.
    block_chain.new_transaction(
        sender="0",
        receiver=node_identifier,
        amount=1,
    )

    # Forge the new Block by adding it to the chain
    block = block_chain.new_block(proof)

    response = {
        'message': "New Block Forged",
        'index': block['index'],
        'transactions': block['transactions'],
        'proof': block['proof'],
        'previous_hash': block['previous_hash'],
    }
    return jsonify(response), 200


@app.route('/transactions/new', methods=['POST'])
def new_transaction():
    values = request.json
    print(values)

    # Check that the required fields are in the POST'ed data
    required = ['sender', 'receiver', 'amount']
    if not all(k in values for k in required):
        return 'Missing values', 400

    # Create a new Transaction
    index = block_chain.new_transaction(values['sender'], values['receiver'], values['amount'])

    # response = {'message': f'Transaction will be added to Block {index}'}
    response = {'message': 'Transaction will be added to Block {0}'.format(index)}
    return jsonify(response), 201


@app.route('/chain', methods=['GET'])
def full_chain():
    response = {
        'chain': block_chain.chain,
        'length': len(block_chain.chain),
    }
    return jsonify(response), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8302)
