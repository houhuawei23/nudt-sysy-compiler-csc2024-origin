a map-like container if you need efficient look-up of a value based on another value. Map-like containers also support efficient queries for containment (whether a key is in the map). Map-like containers generally do not support efficient reverse mapping (values to keys). If you need that, use two maps. Some map-like containers also support efficient iteration through the keys in sorted order. Map-like containers are the most expensive sort, only use them if you need one of these capabilities.

a set-like container if you need to put a bunch of stuff into a container that automatically eliminates duplicates. Some set-like containers support efficient iteration through the elements in sorted order. Set-like containers are more expensive than sequential containers.

a sequential container provides the most efficient way to add elements and keeps track of the order they are added to the collection. They permit duplicates and support efficient iteration, but do not support efficient look-up based on a key.

a string container is a specialized sequential container or reference structure that is used for character or byte arrays.

a bit container provides an efficient way to store and perform set operations on sets of numeric id’s, while automatically eliminating duplicates. Bit containers require a maximum of 1 bit for each identifier you want to store.