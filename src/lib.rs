//! `amalgamator` is a set/map like data structure that allows you to combine members of the set/map together
//! based on some criteria. This is useful when you want to merge objects together that aren't strictly equal,
//! but are similar enough that you want to treat them as the same object.
//!
//! This can be useful in a variety of situations, such as:
//!
//! - Merging data from multiple sources
//! - Entity deduplication
//! - Data normalization
//!
//! Amalgamator is implemented as a set/map like data structure, generic over the type of the elements it contains.
//! Amalgamator uses a user-defined function to determine if two elements should be combined and a user-defined function
//! to merge two elements together. This is the [`Amalgamate`] trait.
//!
//! Lets first take a look at how we might implement `Amalgamate` trait:
//!
//! ```
//! use amalgamator::Amalgamate;
//! use std::collections::HashSet;
//!
//! struct Person {
//!     name: String,
//!     friends: HashSet<String>,
//! }
//!
//! impl Amalgamate for Person {
//!     type Key = String;
//!
//!     fn key(&self) -> Self::Key {
//!         self.name.clone()
//!     }
//!
//!     fn amalgamate(&mut self, other: Self) {
//!         self.friends.extend(other.friends);
//!     }
//! }
//! ```
//!
//! In this example, we have a `Person` struct that has a `name` and a set of `friends`.
//! We implement the `Amalgamate` trait for `Person` by defining the `Key` type as `String` and
//! implementing the `key` and `amalgamate` functions.
//! We assume that the value of the `name` field is unique to each person,
//! so we use it as the key to determine if two `Person` objects should be combined.
//!
//! Now that we have implemented the `Amalgamate` trait, we can use the `Amalgamator` data structure
//! to combine `Person` objects together. Better yet, we can use it just like a regular set/map data structure.
//!
//! ```
//! use amalgamator::Amalgamator;
//! # use amalgamator::Amalgamate;
//! # use std::collections::HashSet;
//! # struct Person {
//! #     name: String,
//! #     friends: HashSet<String>,
//! # }
//! #
//! # impl Amalgamate for Person {
//! #     type Key = String;
//! #
//! #     fn key(&self) -> Self::Key {
//! #         self.name.clone()
//! #     }
//! #
//! #     fn amalgamate(&mut self, other: Self) {
//! #         self.friends.extend(other.friends);
//! #     }
//! # }
//!
//! let mut amalgamator = Amalgamator::new();
//!
//! let alice = Person {
//!     name: "Alice".to_string(),
//!     friends: ["Bob", "Charlie"].iter().map(|s| s.to_string()).collect(),
//! };
//!
//! let bob = Person {
//!     name: "Bob".to_string(),
//!     friends: ["Alice", "Charlie"].iter().map(|s| s.to_string()).collect(),
//! };
//!
//! let other_alice = Person {
//!     name: "Alice".to_string(),
//!     friends: ["David", "Eve"].iter().map(|s| s.to_string()).collect(),
//! };
//!
//! amalgamator.add(alice);
//! amalgamator.add(bob);
//! amalgamator.add(other_alice);
//!
//! assert_eq!(amalgamator.len(), 2);
//!
//! let alice = &amalgamator["Alice"];
//! assert_eq!(alice.friends.len(), 4);
//! ```
//!
//! In this example, we create an `Amalgamator` and add three `Person` objects to it.
//! We then verify that the `Amalgamator` contains only two `Person` objects,
//! as the two `Alice` objects (with the same name) have been combined.
//! We then retrieve the `Alice` object from the `Amalgamator` and verify that it has all the friends from both `Alice` objects.
//!
//! ## Features
//!
//! - `serde`: Enables serialization and deserialization of `Amalgamator` using Serde.

use std::{
    borrow::Borrow,
    collections::hash_map::{HashMap, RandomState},
    hash::{BuildHasher, Hash},
    ops::{Index, IndexMut},
};

/// Describes how an object can be combined with another object based on some criteria.
pub trait Amalgamate {
    /// The type of the key that will be used to determine if two elements should be combined.
    ///
    /// This type must implement `Hash` and `Eq` as it will internally be used as a key in a `HashMap`.
    type Key: Hash + Eq;

    /// Constructs the key that will be used to determine if two elements should be combined.
    fn key(&self) -> Self::Key;

    /// Determines if two elements should be combined.
    fn amalgamate(&mut self, other: Self);
}

/// A set/map like data structure that allows you to combine members of the set/map together based on some criteria.
#[derive(Debug, Clone)]
pub struct Amalgamator<T: Amalgamate, S = RandomState>(HashMap<T::Key, T, S>);

impl<T> Amalgamator<T>
where
    T: Amalgamate,
{
    /// Creates a new `Amalgamator` with an empty set/map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new `Amalgamator` with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self(HashMap::with_capacity(capacity))
    }

    /// Returns the number of elements the map can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    /// Adds an element to the `Amalgamator`.
    ///
    /// If an element with the same key already exists in the `Amalgamator`, the two elements will be combined.
    /// This will be done by calling `Amalgamate::amalgamate` on the existing element with the new element as
    /// an argument.
    pub fn add(&mut self, item: T) {
        let key = item.key();
        if let Some(existing) = self.get_by_key_mut(&key) {
            existing.amalgamate(item);
        } else {
            self.0.insert(key, item);
        }
    }

    /// Returns a reference to the element with the given key.
    pub fn get_by_key<Q>(&self, key: &Q) -> Option<&T>
    where
        T::Key: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.0.get(key)
    }

    /// Returns a mutable reference to the element with the given key.
    pub fn get_by_key_mut<Q>(&mut self, key: &Q) -> Option<&mut T>
    where
        T::Key: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.0.get_mut(key)
    }

    /// Removes the element with the given key from the `Amalgamator` and returns it.
    pub fn remove_by_key<Q>(&mut self, key: &Q) -> Option<T>
    where
        T::Key: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.0.remove(key)
    }

    /// Returns the key-value pair corresponding to the supplied key.
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&T::Key, &T)>
    where
        T::Key: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.0.get_key_value(key)
    }

    /// Removes the element from the `Amalgamator` and returns it.
    pub fn remove(&mut self, item: &T) -> Option<T> {
        self.remove_by_key(&item.key())
    }

    /// Returns the number of elements in the `Amalgamator`.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the `Amalgamator` is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Removes all elements from the `Amalgamator`.
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Returns `true` if the `Amalgamator` contains the given element.
    ///
    /// Keep in mind that this is checking for something that has the same key, not the same value.
    pub fn contains(&self, item: &T) -> bool {
        self.contains_key(&item.key())
    }

    /// Returns `true` if the `Amalgamator` contains an element with the given key.
    ///
    /// Keep in mind that this is checking for something that has the same key, not the same value.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        T::Key: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.0.contains_key(key)
    }

    /// Returns an iterator over the keys of the `Amalgamator`.
    pub fn keys(&self) -> impl Iterator<Item = &T::Key> {
        self.0.keys()
    }

    /// Returns an iterator over the values of the `Amalgamator`.
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.0.values()
    }

    /// Returns an iterator over the mutable values of the `Amalgamator`.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.0.values_mut()
    }

    /// Returns an iterator over the values of the `Amalgamator`.
    pub fn into_values(self) -> impl Iterator<Item = T> {
        self.0.into_values()
    }

    /// Returns an iterator over the keys of the `Amalgamator`.
    pub fn into_keys(self) -> impl Iterator<Item = T::Key> {
        self.0.into_keys()
    }

    /// Returns an iterator over the elements of the `Amalgamator`.
    pub fn iter(&self) -> impl Iterator<Item = (&T::Key, &T)> {
        self.0.iter()
    }

    /// Returns an iterator over the mutable elements of the `Amalgamator`.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&T::Key, &mut T)> {
        self.0.iter_mut()
    }

    /// Retains only the elements that satisfy the predicate.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        self.0.retain(|_, value| f(value));
    }

    /// Removes all elements from the `Amalgamator` and returns them in an iterator.
    pub fn drain(&mut self) -> impl Iterator<Item = T> + '_ {
        self.0.drain().map(|(_, value)| value)
    }
}

impl<T, S> Amalgamator<T, S>
where
    T: Amalgamate,
    S: BuildHasher,
{
    /// Creates a new `Amalgamator` with the given hasher.
    pub fn with_hasher(hash_builder: S) -> Self {
        Self(HashMap::with_hasher(hash_builder))
    }

    /// Creates a new `Amalgamator` with the given hasher and capacity.
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        Self(HashMap::with_capacity_and_hasher(capacity, hash_builder))
    }

    /// Returns a reference to the hasher used by the `Amalgamator`.
    pub fn hasher(&self) -> &S {
        self.0.hasher()
    }

    /// Reserves capacity for at least `additional` more elements to be inserted in the `Amalgamator`.
    pub fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);
    }

    /// Shrinks the capacity of the `Amalgamator` as much as possible.
    pub fn shrink_to_fit(&mut self) {
        self.0.shrink_to_fit();
    }

    /// Shrinks the capacity of the `Amalgamator` to a minimum.
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.0.shrink_to(min_capacity);
    }
}

impl<T> Default for Amalgamator<T>
where
    T: Amalgamate,
{
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<T> Extend<T> for Amalgamator<T>
where
    T: Amalgamate,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for item in iter {
            self.add(item);
        }
    }
}

impl<Q, T> Index<&Q> for Amalgamator<T>
where
    T: Amalgamate,
    T::Key: Borrow<Q>,
    Q: Hash + Eq + ?Sized,
{
    type Output = T;

    fn index(&self, key: &Q) -> &Self::Output {
        self.get_by_key(key).expect("no entry found for key")
    }
}

impl<Q, T> IndexMut<&Q> for Amalgamator<T>
where
    T: Amalgamate,
    T::Key: Borrow<Q>,
    Q: Hash + Eq + ?Sized,
{
    fn index_mut(&mut self, key: &Q) -> &mut Self::Output {
        self.get_by_key_mut(key).expect("no entry found for key")
    }
}

impl<T> FromIterator<T> for Amalgamator<T>
where
    T: Amalgamate,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut amalgamator = Self::default();
        amalgamator.extend(iter);
        amalgamator
    }
}

impl<V, const N: usize> From<[V; N]> for Amalgamator<V>
where
    V: Amalgamate,
{
    fn from(array: [V; N]) -> Self {
        Self::from_iter(array)
    }
}

impl<T> IntoIterator for Amalgamator<T>
where
    T: Amalgamate,
{
    type Item = (T::Key, T);
    type IntoIter = std::collections::hash_map::IntoIter<T::Key, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a Amalgamator<T>
where
    T: Amalgamate,
{
    type Item = (&'a T::Key, &'a T);
    type IntoIter = std::collections::hash_map::Iter<'a, T::Key, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Amalgamator<T>
where
    T: Amalgamate,
{
    type Item = (&'a T::Key, &'a mut T);
    type IntoIter = std::collections::hash_map::IterMut<'a, T::Key, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

impl<T> PartialEq for Amalgamator<T>
where
    T: Amalgamate + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for Amalgamator<T> where T: Amalgamate + Eq {}

#[cfg(feature = "serde")]
mod serde {
    use super::*;
    use ::serde::{Deserialize, Deserializer, Serialize, Serializer};

    impl<T> Serialize for Amalgamator<T>
    where
        T: Amalgamate + Serialize,
        T::Key: Serialize,
    {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            self.0.serialize(serializer)
        }
    }

    impl<'a, T> Deserialize<'a> for Amalgamator<T>
    where
        T: Amalgamate + Deserialize<'a>,
        T::Key: Deserialize<'a>,
    {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'a>,
        {
            let map = HashMap::<T::Key, T>::deserialize(deserializer)?;
            Ok(Self(map))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashSet;

    #[cfg(feature = "serde")]
    use ::serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Eq)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    struct Person {
        name: String,
        friends: HashSet<String>,
    }

    impl Person {
        fn new(name: &str, friends: &[&str]) -> Self {
            Self {
                name: name.to_string(),
                friends: friends.iter().map(|s| s.to_string()).collect(),
            }
        }
    }

    impl Amalgamate for Person {
        type Key = String;

        fn key(&self) -> Self::Key {
            self.name.clone()
        }

        fn amalgamate(&mut self, other: Self) {
            self.friends.extend(other.friends);
        }
    }

    #[test]
    fn test_add() {
        let mut amalgamator = Amalgamator::new();

        let alice = Person::new("Alice", &["Bob", "Charlie"]);
        let bob = Person::new("Bob", &["Alice", "Charlie"]);
        let other_alice = Person::new("Alice", &["David", "Eve"]);

        amalgamator.add(alice.clone());
        amalgamator.add(bob.clone());
        amalgamator.add(other_alice.clone());

        assert_eq!(amalgamator.len(), 2);

        let alice = &amalgamator["Alice"];
        assert_eq!(alice.friends.len(), 4);
    }

    #[test]
    fn test_with_capacity() {
        let amalgamator: Amalgamator<Person> = Amalgamator::with_capacity(10);
        assert!(amalgamator.capacity() >= 10);
    }

    #[test]
    fn test_with_capacity_and_hasher() {
        let amalgamator: Amalgamator<Person> =
            Amalgamator::with_capacity_and_hasher(10, RandomState::new());
        assert!(amalgamator.capacity() >= 10);
    }

    #[test]
    fn test_get_by_key_present() {
        let mut amalgamator = Amalgamator::new();
        amalgamator.add(Person::new("Alice", &["Bob", "Charlie"]));
        assert!(amalgamator.get_by_key("Alice").is_some());
    }

    #[test]
    fn test_get_by_key_absent() {
        let amalgamator: Amalgamator<Person> = Amalgamator::new();
        assert!(amalgamator.get_by_key("Alice").is_none());
    }

    #[test]
    fn test_get_by_key_mut_present() {
        let mut amalgamator = Amalgamator::new();
        amalgamator.add(Person::new("Alice", &["Bob", "Charlie"]));
        assert!(amalgamator.get_by_key_mut("Alice").is_some());
    }

    #[test]
    fn test_get_by_key_mut_absent() {
        let mut amalgamator: Amalgamator<Person> = Amalgamator::new();
        assert!(amalgamator.get_by_key_mut("Alice").is_none());
    }

    #[test]
    fn test_remove_by_key_present() {
        let mut amalgamator = Amalgamator::new();
        amalgamator.add(Person::new("Alice", &["Bob", "Charlie"]));
        assert!(amalgamator.remove_by_key("Alice").is_some());
        assert_eq!(amalgamator.len(), 0);
    }

    #[test]
    fn test_remove_by_key_absent() {
        let mut amalgamator = Amalgamator::new();
        amalgamator.add(Person::new("Bob", &["Alic", "Charlie"]));
        assert!(amalgamator.remove_by_key("Alice").is_none());
    }

    #[test]
    fn test_get_key_value_present() {
        let mut amalgamator = Amalgamator::new();
        amalgamator.add(Person::new("Alice", &["Bob", "Charlie"]));
        assert!(amalgamator.get_key_value("Alice").is_some());
    }

    #[test]
    fn test_get_key_value_absent() {
        let amalgamator: Amalgamator<Person> = Amalgamator::new();
        assert!(amalgamator.get_key_value("Alice").is_none());
    }

    #[test]
    fn test_remove_present() {
        let mut amalgamator = Amalgamator::new();
        let alice = Person::new("Alice", &["Bob", "Charlie"]);
        amalgamator.add(alice.clone());
        assert!(amalgamator.remove(&alice).is_some());
        assert_eq!(amalgamator.len(), 0);
    }

    #[test]
    fn test_remove_absent() {
        let mut amalgamator = Amalgamator::new();
        let alice = Person::new("Alice", &["Bob", "Charlie"]);
        amalgamator.add(alice.clone());
        let bob = Person::new("Bob", &["Alice", "Charlie"]);
        assert!(amalgamator.remove(&bob).is_none());
        assert_eq!(amalgamator.len(), 1);
    }

    #[test]
    fn test_contains_present() {
        let mut amalgamator = Amalgamator::new();
        let alice = Person::new("Alice", &["Bob", "Charlie"]);
        amalgamator.add(alice.clone());
        assert!(amalgamator.contains(&alice));
    }

    #[test]
    fn test_contains_absent() {
        let mut amalgamator = Amalgamator::new();
        let alice = Person::new("Alice", &["Bob", "Charlie"]);
        let bob = Person::new("Bob", &["Alice", "Charlie"]);
        amalgamator.add(alice.clone());
        assert!(!amalgamator.contains(&bob));
    }

    #[test]
    fn test_contains_key_present() {
        let mut amalgamator = Amalgamator::new();
        let alice = Person::new("Alice", &["Bob", "Charlie"]);
        amalgamator.add(alice.clone());
        assert!(amalgamator.contains_key("Alice"));
    }

    #[test]
    fn test_contains_key_absent() {
        let amalgamator: Amalgamator<Person> = Amalgamator::new();
        assert!(!amalgamator.contains_key("Alice"));
    }

    #[test]
    fn test_keys() {
        let mut amalgamator = Amalgamator::new();
        amalgamator.add(Person::new("Alice", &["Bob", "Charlie"]));
        amalgamator.add(Person::new("Bob", &["Alice", "Charlie"]));
        assert_eq!(
            amalgamator
                .keys()
                .map(String::as_str)
                .collect::<HashSet<_>>(),
            HashSet::from(["Alice", "Bob"])
        );
    }

    #[test]
    fn test_values() {
        let mut amalgamator = Amalgamator::new();
        amalgamator.add(Person::new("Alice", &["Bob", "Charlie"]));
        amalgamator.add(Person::new("Bob", &["Alice", "Charlie"]));
        assert_eq!(
            amalgamator
                .values()
                .map(|p| p.name.clone())
                .collect::<HashSet<_>>(),
            HashSet::from(["Alice".to_owned(), "Bob".to_owned()])
        );
    }

    #[test]
    fn test_values_mut() {
        let mut amalgamator = Amalgamator::new();
        amalgamator.add(Person::new("Alice", &["Bob", "Charlie"]));
        amalgamator.add(Person::new("Bob", &["Alice", "Charlie"]));
        for person in amalgamator.values_mut() {
            person.friends.clear();
        }
        assert!(amalgamator.values().all(|p| p.friends.is_empty()));
    }

    #[test]
    fn test_into_values() {
        let mut amalgamator = Amalgamator::new();
        amalgamator.add(Person::new("Alice", &["Bob", "Charlie"]));
        amalgamator.add(Person::new("Bob", &["Alice", "Charlie"]));
        assert_eq!(
            amalgamator
                .into_values()
                .map(|p| p.name)
                .collect::<HashSet<_>>(),
            HashSet::from(["Alice".to_owned(), "Bob".to_owned()])
        );
    }

    #[test]
    fn test_into_keys() {
        let mut amalgamator = Amalgamator::new();
        amalgamator.add(Person::new("Alice", &["Bob", "Charlie"]));
        amalgamator.add(Person::new("Bob", &["Alice", "Charlie"]));
        assert_eq!(
            amalgamator.into_keys().collect::<HashSet<_>>(),
            HashSet::from(["Alice".to_owned(), "Bob".to_owned()])
        );
    }

    #[test]
    fn test_iter() {
        let mut amalgamator = Amalgamator::new();
        amalgamator.add(Person::new("Alice", &["Bob", "Charlie"]));
        amalgamator.add(Person::new("Bob", &["Alice", "Charlie"]));
        assert_eq!(
            amalgamator
                .iter()
                .map(|(k, v)| (k.as_str(), v.name.as_str()))
                .collect::<HashSet<_>>(),
            HashSet::from([("Alice", "Alice"), ("Bob", "Bob")])
        );
    }

    #[test]
    fn test_iter_mut() {
        let mut amalgamator = Amalgamator::new();
        amalgamator.add(Person::new("Alice", &["Bob", "Charlie"]));
        amalgamator.add(Person::new("Bob", &["Alice", "Charlie"]));
        for (_, person) in amalgamator.iter_mut() {
            person.friends.clear();
        }
        assert!(amalgamator.values().all(|p| p.friends.is_empty()));
    }

    #[test]
    fn test_retain() {
        let mut amalgamator = Amalgamator::new();
        amalgamator.add(Person::new("Alice", &["Bob", "Charlie"]));
        amalgamator.add(Person::new("Bob", &["Alice", "Charlie"]));
        amalgamator.retain(|person| person.name == "Alice");
        assert_eq!(amalgamator.len(), 1);
        assert_eq!(amalgamator.keys().collect::<Vec<_>>(), ["Alice"]);
    }

    #[test]
    fn test_drain() {
        let mut amalgamator = Amalgamator::new();
        amalgamator.add(Person::new("Alice", &["Bob", "Charlie"]));
        amalgamator.add(Person::new("Bob", &["Alice", "Charlie"]));
        let drained: Vec<_> = amalgamator.drain().collect();
        assert_eq!(drained.len(), 2);
        assert_eq!(amalgamator.len(), 0);
    }

    #[test]
    fn test_extend() {
        let mut amalgamator = Amalgamator::new();
        let people = vec![
            Person::new("Alice", &["Bob", "Charlie"]),
            Person::new("Bob", &["Alice", "Charlie"]),
        ];
        amalgamator.extend(people);
        assert_eq!(amalgamator.len(), 2);
    }

    #[test]
    fn test_from_iter() {
        let people = vec![
            Person::new("Alice", &["Bob", "Charlie"]),
            Person::new("Bob", &["Alice", "Charlie"]),
        ];
        let amalgamator: Amalgamator<Person> = people.into_iter().collect();
        assert_eq!(amalgamator.len(), 2);
    }

    #[test]
    fn test_from_array() {
        let people = [
            Person::new("Alice", &["Bob", "Charlie"]),
            Person::new("Bob", &["Alice", "Charlie"]),
        ];
        let amalgamator: Amalgamator<Person> = people.into();
        assert_eq!(amalgamator.len(), 2);
    }

    #[test]
    fn test_into_iter() {
        let mut amalgamator = Amalgamator::new();
        amalgamator.add(Person::new("Alice", &["Bob", "Charlie"]));
        amalgamator.add(Person::new("Bob", &["Alice", "Charlie"]));
        let items: Vec<_> = amalgamator.into_iter().collect();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_into_iter_ref() {
        let mut amalgamator = Amalgamator::new();
        amalgamator.add(Person::new("Alice", &["Bob", "Charlie"]));
        amalgamator.add(Person::new("Bob", &["Alice", "Charlie"]));
        let items: Vec<_> = amalgamator.iter().collect();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_into_iter_mut() {
        let mut amalgamator = Amalgamator::new();
        amalgamator.add(Person::new("Alice", &["Bob", "Charlie"]));
        amalgamator.add(Person::new("Bob", &["Alice", "Charlie"]));
        for (_, person) in amalgamator.iter_mut() {
            person.friends.clear();
        }
        assert!(amalgamator.values().all(|p| p.friends.is_empty()));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde() {
        let mut amalgamator = Amalgamator::new();
        amalgamator.add(Person::new("Alice", &["Bob", "Charlie"]));
        amalgamator.add(Person::new("Bob", &["Alice", "Charlie"]));

        let serialized = serde_json::to_string(&amalgamator).unwrap();
        let deserialized: Amalgamator<Person> = serde_json::from_str(&serialized).unwrap();

        assert_eq!(amalgamator, deserialized);
    }
}
