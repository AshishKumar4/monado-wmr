namespace UnifiedAssociator

universe keyUniverse

structure FiniteKeys (keyType : Type keyUniverse) where
  keys : List keyType
  nodup : keys.Nodup

def FiniteKeys.Contains (domain : FiniteKeys keyType) (key : keyType) : Prop :=
  key ∈ domain.keys

structure BlobKey where
  camera : Nat
  slot : Nat
deriving DecidableEq, Repr

structure LedKey where
  device : Nat
  led : Nat
deriving DecidableEq, Repr

structure Hypothesis where
  id : Nat
deriving DecidableEq, Repr

end UnifiedAssociator
