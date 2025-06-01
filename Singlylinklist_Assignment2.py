class Node:
    """Class representing a single node in a singly linked list."""
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    """Class for managing the singly linked list."""
    def __init__(self):
        self.head = None

    def add_node(self, data):
        """Add a node with the given data to the end of the list."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            print(f"Added head node with value {data}")
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
            print(f"Added node with value {data}")

    def print_list(self):
        """Print all elements in the list."""
        if not self.head:
            print("The list is empty.")
            return
        current = self.head
        print("Linked List:", end=" ")
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def delete_nth_node(self, n):
        """Delete the nth node (1-based index)."""
        try:
            if not self.head:
                raise IndexError("Cannot delete from an empty list.")
            if n <= 0:
                raise IndexError("Index must be 1 or greater.")

            if n == 1:
                print(f"Deleted node at index {n} with value {self.head.data}")
                self.head = self.head.next
                return

            current = self.head
            for i in range(n - 2):
                if current.next is None:
                    raise IndexError("Index out of range.")
                current = current.next

            if current.next is None:
                raise IndexError("Index out of range.")

            print(f"Deleted node at index {n} with value {current.next.data}")
            current.next = current.next.next

        except IndexError as e:
            print(f"Error: {e}")


# Test the LinkedList class

if __name__ == "__main__":
    ll = LinkedList()
    ll.print_list()  

    # Add sample nodes
    ll.add_node(10)
    ll.add_node(20)
    ll.add_node(30)
    ll.add_node(40)
    ll.print_list()  # 10 -> 20 -> 30 -> 40 -> None

    # Delete a node with a valid index
    ll.delete_nth_node(3)  # Should delete node with value 30
    ll.print_list()        # 10 -> 20 -> 40 -> None

    # Try deleting from an out-of-range index
    ll.delete_nth_node(10)  # Should raise "Index out of range"

    # Try deleting from an empty list
    empty_list = LinkedList()
    empty_list.delete_nth_node(1)  # Should raise "Cannot delete from an empty list"
