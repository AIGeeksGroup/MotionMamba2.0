// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export function getModalChoiceSaved(saveChoiceKey: string): boolean {
  try {
    if (typeof window === "undefined") return false;
    return window.localStorage.getItem(`viser.modalChoice.${saveChoiceKey}`) === "1";
  } catch {
    return false;
  }
}

export function setModalChoiceSaved(saveChoiceKey: string): void {
  try {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(`viser.modalChoice.${saveChoiceKey}`, "1");
  } catch {
    // Ignore (e.g. storage disabled).
  }
}

