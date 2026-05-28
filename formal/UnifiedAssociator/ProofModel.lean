import UnifiedAssociator.Keys

namespace UnifiedAssociator

structure Assignment where
  ledOfBlob : BlobKey → Option LedKey

structure ScoreModel where
  reprojectionCost : Hypothesis → Nat
  priorCost : Hypothesis → Nat
  jointCost : Hypothesis → Nat
  jointCostEq :
    ∀ candidate, jointCost candidate = reprojectionCost candidate + priorCost candidate

structure Frame where
  blobKeys : FiniteKeys BlobKey
  ledKeys : FiniteKeys LedKey
  hypotheses : FiniteKeys Hypothesis
  hypothesesNonempty : hypotheses.keys ≠ []
  assignment : Hypothesis → Assignment
  compatible : Hypothesis → BlobKey → LedKey → Prop
  score : ScoreModel

def AssignmentInjective (assignment : Assignment) : Prop :=
  ∀ firstBlob secondBlob ledKey,
    assignment.ledOfBlob firstBlob = some ledKey →
      assignment.ledOfBlob secondBlob = some ledKey →
        firstBlob = secondBlob

def AssignmentUsesFrameKeys (frame : Frame) (assignment : Assignment) : Prop :=
  (∀ blobKey ledKey,
      assignment.ledOfBlob blobKey = some ledKey →
        blobKey ∈ frame.blobKeys.keys ∧ ledKey ∈ frame.ledKeys.keys) ∧
    (∀ blobKey,
      blobKey ∉ frame.blobKeys.keys →
        assignment.ledOfBlob blobKey = none)

def AssignmentCompatible (frame : Frame) (candidate : Hypothesis) : Prop :=
  let assignment := frame.assignment candidate
  AssignmentUsesFrameKeys frame assignment ∧
    AssignmentInjective assignment ∧
      ∀ blobKey ledKey,
        assignment.ledOfBlob blobKey = some ledKey →
          frame.compatible candidate blobKey ledKey

def HypothesisWellFormed (frame : Frame) (candidate : Hypothesis) : Prop :=
  candidate ∈ frame.hypotheses.keys ∧ AssignmentCompatible frame candidate

def IsJointArgMin (frame : Frame) (candidate : Hypothesis) : Prop :=
  candidate ∈ frame.hypotheses.keys ∧
    ∀ challenger,
      challenger ∈ frame.hypotheses.keys →
        frame.score.jointCost candidate ≤ frame.score.jointCost challenger

inductive StateDecision where
  | reject
  | accept (winner : Hypothesis)
  | reset (winner : Hypothesis)
deriving DecidableEq, Repr

namespace StateDecision

def selected : StateDecision → Option Hypothesis
  | .reject => none
  | .accept winner => some winner
  | .reset winner => some winner

def HasPose : StateDecision → Prop
  | .reject => False
  | .accept _ => True
  | .reset _ => True

end StateDecision

structure DecisionPolicy where
  accepts : Hypothesis → Prop
  resets : Hypothesis → Prop

def DecisionSound (frame : Frame) (policy : DecisionPolicy) : StateDecision → Prop
  | .reject =>
      ∀ candidate, IsJointArgMin frame candidate → ¬ policy.accepts candidate
  | .accept winner =>
      IsJointArgMin frame winner ∧ policy.accepts winner ∧ ¬ policy.resets winner
  | .reset winner =>
      IsJointArgMin frame winner ∧ policy.accepts winner ∧ policy.resets winner

def SelectedArgMin (frame : Frame) : StateDecision → Prop
  | .reject => True
  | .accept winner => IsJointArgMin frame winner
  | .reset winner => IsJointArgMin frame winner

inductive TelemetryStream where
  | frameSummary
  | hypothesisScore
  | compatibilityGate
  | stateDecision
  | residual
deriving DecidableEq, Repr

structure TelemetryObligations (frame : Frame) (decision : StateDecision) where
  emits : TelemetryStream → Prop
  frameSummary : emits TelemetryStream.frameSummary
  hypothesisScores :
    ∀ candidate, candidate ∈ frame.hypotheses.keys → emits TelemetryStream.hypothesisScore
  compatibilityGates : emits TelemetryStream.compatibilityGate
  decisionEvent : emits TelemetryStream.stateDecision
  residualWhenPose :
    StateDecision.HasPose decision → emits TelemetryStream.residual

structure ProofObligations (frame : Frame) (policy : DecisionPolicy) (decision : StateDecision) where
  allHypothesesWellFormed :
    ∀ candidate, candidate ∈ frame.hypotheses.keys → AssignmentCompatible frame candidate
  decisionSound : DecisionSound frame policy decision
  telemetry : TelemetryObligations frame decision

theorem decision_sound_selected_argmin
    {frame : Frame}
    {policy : DecisionPolicy}
    {decision : StateDecision} :
    DecisionSound frame policy decision → SelectedArgMin frame decision := by
  intro soundDecision
  cases decision with
  | reject =>
      exact True.intro
  | accept winner =>
      exact soundDecision.left
  | reset winner =>
      exact soundDecision.left

theorem proof_obligations_selected_argmin
    {frame : Frame}
    {policy : DecisionPolicy}
    {decision : StateDecision}
    (obligations : ProofObligations frame policy decision) :
    SelectedArgMin frame decision :=
  decision_sound_selected_argmin obligations.decisionSound

theorem proof_obligations_assignments_compatible
    {frame : Frame}
    {policy : DecisionPolicy}
    {decision : StateDecision}
    (obligations : ProofObligations frame policy decision)
    {candidate : Hypothesis} :
    candidate ∈ frame.hypotheses.keys → AssignmentCompatible frame candidate := by
  intro candidateInFrame
  exact obligations.allHypothesesWellFormed candidate candidateInFrame

theorem proof_obligations_have_decision_telemetry
    {frame : Frame}
    {policy : DecisionPolicy}
    {decision : StateDecision}
    (obligations : ProofObligations frame policy decision) :
    obligations.telemetry.emits TelemetryStream.stateDecision :=
  obligations.telemetry.decisionEvent

end UnifiedAssociator
