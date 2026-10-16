// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { GuiButtonMessage } from "../WebsocketMessages";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { Box } from "@mantine/core";

import { Button } from "@mantine/core";
import React from "react";
import { htmlIconWrapper } from "./ComponentStyles.css";
import { toMantineColor } from "./colorUtils";
import { ViewerContext } from "../ViewerContext";
import { setModalChoiceSaved } from "../QuickStartPersistence";

export default function ButtonComponent({
  uuid,
  container_uuid,
  props: { visible, disabled, label, color, _icon_html: icon_html },
}: GuiButtonMessage) {
  const { messageSender } = React.useContext(GuiComponentContext)!;
  const viewer = React.useContext(ViewerContext)!;
  if (!(visible ?? true)) return null;

  const modalSaveChoiceKey = viewer.useGui((state) => {
    const modal = state.modals.find((m) => m.uuid === container_uuid);
    return modal?.save_choice ?? null;
  });

  return (
    <Box mx="xs" pb="0.5em">
      <Button
        id={uuid}
        fullWidth
        color={toMantineColor(color)}
        onClick={() => {
          // Only persist a modal's acknowledgement on explicit button clicks.
          // (Do NOT persist when the modal is closed via the red X.)
          if (modalSaveChoiceKey) {
            setModalChoiceSaved(modalSaveChoiceKey);
          }
          messageSender({
            type: "GuiUpdateMessage",
            uuid: uuid,
            updates: { value: true },
          });
        }}
        style={{
          height: "2em",
        }}
        disabled={disabled ?? false}
        size="sm"
        leftSection={
          icon_html === null ? undefined : (
            <div
              className={htmlIconWrapper}
              dangerouslySetInnerHTML={{ __html: icon_html }}
            />
          )
        }
      >
        {label}
      </Button>
    </Box>
  );
}
