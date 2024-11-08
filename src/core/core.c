#include "conv.h"
#include "core.h"

// static void reverse_nodes(struct device_node *parent)
// {
// 	struct device_node *child, *next;

// 	/* In-depth first */
// 	child = parent->child;
// 	while (child) {
// 		reverse_nodes(child);

// 		child = child->sibling;
// 	}

// 	/* Reverse the nodes in the child list */
// 	child = parent->child;
// 	parent->child = NULL;
// 	while (child) {
// 		next = child->sibling;

// 		child->sibling = parent->child;
// 		parent->child = child;
// 		child = next;
// 	}
// }