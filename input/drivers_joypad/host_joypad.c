/*  RetroArch - A frontend for libretro.
 *  Copyright (C) 2026 - Aphrody
 *
 *  RetroArch is free software: you can redistribute it and/or modify it under the terms
 *  of the GNU General Public License as published by the Free Software Found-
 *  ation, either version 3 of the License, or (at your option) any later version.
 *
 *  RetroArch is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 *  PURPOSE.  See the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along with RetroArch.
 *  If not, see <http://www.gnu.org/licenses/>.
 */

/* Pads the libretroarch host fills in. Buttons are RetroPad bits, so
 * the built-in "Host Gamepad" profile binds each to itself. */

#include <stdint.h>

#include <boolean.h>
#include <libretro.h>

#include "../input_driver.h"
#include "../../tasks/tasks_internal.h"
#include "../../frontend/libretroarch.h"

#define HOST_JOYPAD_AXES    6
#define HOST_JOYPAD_NAME    "Host Gamepad"

typedef struct
{
   uint32_t buttons;
   int16_t axes[HOST_JOYPAD_AXES];
   bool connected;
} host_joypad_data_t;

static host_joypad_data_t host_joypads[DEFAULT_MAX_PADS];
static bool host_joypad_inited = false;

static void *host_joypad_init(void *data)
{
   unsigned i;
   host_joypad_inited = true;
   for (i = 0; i < DEFAULT_MAX_PADS; i++)
      if (host_joypads[i].connected)
         input_autoconfigure_connect(HOST_JOYPAD_NAME, NULL, NULL,
               "host", i, 0, 0);
   return (void*)-1;
}

static bool host_joypad_query_pad(unsigned pad)
{
   return pad < DEFAULT_MAX_PADS && host_joypads[pad].connected;
}

static void host_joypad_destroy(void)
{
   host_joypad_inited = false;
}

static int32_t host_joypad_button(unsigned port, uint16_t joykey)
{
   if (port >= DEFAULT_MAX_PADS || joykey >= 32)
      return 0;
   return (host_joypads[port].buttons >> joykey) & 1;
}

static int16_t host_joypad_axis(unsigned port, uint32_t joyaxis)
{
   int16_t val;
   if (port >= DEFAULT_MAX_PADS)
      return 0;
   if (AXIS_NEG_GET(joyaxis) < HOST_JOYPAD_AXES)
   {
      val = host_joypads[port].axes[AXIS_NEG_GET(joyaxis)];
      if (val < 0)
         return val;
   }
   else if (AXIS_POS_GET(joyaxis) < HOST_JOYPAD_AXES)
   {
      val = host_joypads[port].axes[AXIS_POS_GET(joyaxis)];
      if (val > 0)
         return val;
   }
   return 0;
}

static int16_t host_joypad_state(
      rarch_joypad_info_t *joypad_info,
      const uint16_t *joykeys,
      const uint32_t *joyaxes,
      unsigned port)
{
   unsigned i;
   int16_t ret       = 0;
   uint16_t port_idx = joypad_info->joy_idx;

   if (port_idx >= DEFAULT_MAX_PADS)
      return 0;
   for (i = 0; i < RARCH_FIRST_CUSTOM_BIND; i++)
      if (     joykeys[i] != NO_BTN
            && host_joypad_button(port_idx, joykeys[i]))
         ret |= (1 << i);
   return ret;
}

static void host_joypad_poll(void) { }

static const char *host_joypad_name(unsigned pad)
{
   return HOST_JOYPAD_NAME;
}

void libretroarch_joypad(unsigned port, uint32_t buttons,
      const int16_t *axes, unsigned num_axes)
{
   host_joypad_data_t *pad;
   unsigned i;

   if (port >= DEFAULT_MAX_PADS)
      return;
   pad          = &host_joypads[port];
   pad->buttons = buttons;
   if (num_axes > HOST_JOYPAD_AXES)
      num_axes = HOST_JOYPAD_AXES;
   for (i = 0; i < num_axes; i++)
      pad->axes[i] = axes[i];
   for (; i < HOST_JOYPAD_AXES; i++)
      pad->axes[i] = 0;
   if (!pad->connected)
   {
      pad->connected = true;
      if (host_joypad_inited)
         input_autoconfigure_connect(HOST_JOYPAD_NAME, NULL, NULL,
               "host", port, 0, 0);
   }
}

input_device_driver_t host_joypad = {
   host_joypad_init,
   host_joypad_query_pad,
   host_joypad_destroy,
   host_joypad_button,
   host_joypad_state,
   NULL, /* get_buttons */
   host_joypad_axis,
   host_joypad_poll,
   NULL, /* set_rumble */
   NULL, /* set_rumble_gain */
   NULL, /* set_sensor_state */
   NULL, /* get_sensor_input */
   host_joypad_name,
   "host",
};
