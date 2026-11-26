// --probe-pads: players and their gamepads (input/input.c) with four virtual SDL gamepads, without the game: each one's
// button reaches its player only, input_pad_swap makes it reach the other (players 1 and 2, then 2 and 4), twice puts it
// back, a gamepad goes to a player without one, player 1 keeps the keyboard with MKW_PAD_P2 (no swap), and bad indices do
// nothing. Why: which controller is which player is decided by the order they connect, and the menu's Controls tab
// changes it. Gamepads connected to the machine come first and only shift the players checked.
// Steam Input (#882): a fifth virtual gamepad with a DualSense's ids and a gyro, listed in SDL_GAMECONTROLLER_IGNORE_DEVICES
// as Steam does, is no player's, and its motion feeds player 1, whose gamepad has none.
#include <SDL3/SDL.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "cfg.h"
#include "host.h"
#include "input.h"
#include "platform/platform.h"

static int failures;
#define CHECK(cond, ...)                                                                                               \
    do {                                                                                                               \
        int ok_ = (cond);                                                                                              \
        printf("%s: ", ok_ ? "ok  " : "FAIL");                                                                         \
        printf(__VA_ARGS__);                                                                                           \
        printf("\n");                                                                                                  \
        failures += !ok_;                                                                                              \
    } while (0)

enum { PADS = 4 };

static SDL_Joystick *vj[PADS];

static uint64_t held(int player) {
    uint64_t b = 0;
    int32_t x[4];
    input_poll_player(player, &b, &x[0], &x[1], &x[2], &x[3]);
    return b;
}

// The player (from base, else -1 none, -2 several) that gets the button of virtual gamepad i held.
static int who(int i, int base) {
    SDL_SetJoystickVirtualButton(vj[i], SDL_GAMEPAD_BUTTON_SOUTH, true);
    SDL_UpdateJoysticks();
    int found = -1;
    for (int p = base; p < INPUT_MAX_PLAYERS; p++)
        if (held(p)) found = found == -1 ? p : -2;
    SDL_SetJoystickVirtualButton(vj[i], SDL_GAMEPAD_BUTTON_SOUTH, false);
    SDL_UpdateJoysticks();
    return found;
}

// The controller Steam hides (#882): attached after the gamepads started, as a DualSense with a gyro and an
// accelerometer. Its button reaches no player, and with no gamepad of the machine before the virtual ones, its motion
// (lying flat, a quarter turn a second about the vertical) is what player 1 reads.
static void steam_hidden_check(int base) {
    const SDL_VirtualJoystickSensorDesc sens[2] = {{SDL_SENSOR_GYRO, 250.0f}, {SDL_SENSOR_ACCEL, 250.0f}};
    SDL_VirtualJoystickDesc d;
    SDL_INIT_INTERFACE(&d);
    d.type = SDL_JOYSTICK_TYPE_GAMEPAD;
    d.vendor_id = 0x054c;
    d.product_id = 0x0ce6;
    d.naxes = SDL_GAMEPAD_AXIS_COUNT;
    d.nbuttons = SDL_GAMEPAD_BUTTON_COUNT;
    d.nsensors = 2;
    d.sensors = sens;
    d.name = "hidden DualSense";
    SDL_JoystickID id = SDL_AttachVirtualJoystick(&d);
    SDL_Joystick *j = id ? SDL_OpenJoystick(id) : NULL;
    CHECK(j != NULL, "Steam Input: the controller Steam hides is listed (%s)", j ? "attached" : SDL_GetError());
    if (!j) return;
    uint64_t b = 0;
    int32_t x[4];
    plat_wait_ns(1100000000ull);   // past RESCAN_MS: the next poll looks for new gamepads
    input_poll(&b, &x[0], &x[1], &x[2], &x[3]);
    SDL_SetJoystickVirtualButton(j, SDL_GAMEPAD_BUTTON_SOUTH, true);
    SDL_UpdateJoysticks();
    int any = 0;
    for (int p = 0; p < INPUT_MAX_PLAYERS; p++) any |= held(p) != 0;
    SDL_SetJoystickVirtualButton(j, SDL_GAMEPAD_BUTTON_SOUTH, false);
    SDL_UpdateJoysticks();
    CHECK(!any, "Steam Input: it is no player's (Steam's virtual gamepad plays for it)");
    if (!base) {
        input_motion_start();
        const float g[3] = {0, 1.5707963f, 0}, a[3] = {0, SDL_STANDARD_GRAVITY, 0};
        for (int i = 0; i < 400; i++) {
            uint64_t t = SDL_GetTicksNS();
            SDL_SendJoystickVirtualSensorData(j, SDL_SENSOR_GYRO, t, g, 3);
            SDL_SendJoystickVirtualSensorData(j, SDL_SENSOR_ACCEL, t, a, 3);
            plat_wait_ns(4000000);
        }
        MotionState st;
        int n = motion_read(motion_player(0), &st, 1, kern_now_ns());
        CHECK(n == 1 && fabsf(st.gyro[2] - 0.25f) < 0.01f && fabsf(st.acc[2] + 1) < 0.05f,
              "Steam Input: its motion feeds player 1 (rate z %.3f rev/s, acc z %.2f G)", n ? st.gyro[2] : 0.0f,
              n ? st.acc[2] : 0.0f);
    }
    SDL_CloseJoystick(j);
    SDL_DetachVirtualJoystick(id);
}

int probe_pads(void) {
    plat_setenv("MKW_VIRTUAL_PADS", "4");
    plat_setenv("MKW_PAD_P2", "0");
    plat_setenv("MKW_KEYBOARD_MOUSE", "0");
    plat_setenv("SDL_GAMECONTROLLER_IGNORE_DEVICES", "0x28de/0x1205,0x054c/0x0ce6");   // as Steam sets it (Deck, DualSense)
    (void)input_player_mask();   // starts the gamepads
    int n = 0, k = 0, base = 0;
    SDL_JoystickID *ids = SDL_GetGamepads(&n);
    for (int i = 0; i < n; i++) {
        if (SDL_IsJoystickVirtual(ids[i]) && k < PADS) vj[k++] = SDL_OpenJoystick(ids[i]);
        else if (!k) base++;
    }
    SDL_free(ids);
    if (k < PADS || base + PADS > INPUT_MAX_PLAYERS) {
        printf("probe-pads: %d virtual gamepads after %d others: not enough to check\n", k, base);
        return 1;
    }
    printf("probe-pads: %d gamepad(s) of the machine first\n", base);
    for (int i = 0; i < PADS; i++) CHECK(who(i, 0) == base + i, "gamepad %d is player %d's only", i + 1, base + i + 1);
    CHECK(input_pad_swap(base, base + 1), "players %d and %d trade", base + 1, base + 2);
    CHECK(who(0, 0) == base + 1 && who(1, 0) == base, "their gamepads follow");
    CHECK(who(2, 0) == base + 2 && who(3, 0) == base + 3, "the others stay");
    CHECK(input_pad_swap(base + 1, base + 3) && who(0, 0) == base + 3 && who(3, 0) == base + 1, "players %d and %d trade", base + 2, base + 4);
    CHECK(input_pad_swap(base + 1, base + 3) && input_pad_swap(base, base + 1), "traded back");
    for (int i = 0; i < PADS; i++) CHECK(who(i, 0) == base + i, "gamepad %d is player %d's again", i + 1, base + i + 1);
    if (base + PADS < INPUT_MAX_PLAYERS) {
        CHECK(input_pad_swap(base + 1, base + PADS) && who(1, 0) == base + PADS, "a gamepad goes to a player without one");
        CHECK(!input_pad_can_move_up(base + 1), "no row to move for the player left without one");
        CHECK(input_pad_swap(base + 1, base + PADS), "and back");
    }
    CHECK(input_pad_can_move_up(base + 1) && !input_pad_can_move_up(0), "player 1 cannot move up, the next can");
    CHECK(!input_pad_swap(0, 0) && !input_pad_swap(-1, 1) && !input_pad_swap(0, INPUT_MAX_PLAYERS), "bad indices do nothing");
    CHECK(!input_pad_swap(INPUT_MAX_PLAYERS - 2, INPUT_MAX_PLAYERS - 1) || base + PADS > INPUT_MAX_PLAYERS - 2,
          "two players without a gamepad do nothing");
    if (!base) {
        // The menu's release wait (#582): a button held when it closes stays ignored until let go, the others work.
        uint64_t b = 0;
        int32_t x[4];
        SDL_SetJoystickVirtualButton(vj[0], SDL_GAMEPAD_BUTTON_SOUTH, true);
        SDL_UpdateJoysticks();
        input_poll(&b, &x[0], &x[1], &x[2], &x[3]);
        CHECK(b != 0, "a held button reaches the game");
        input_menu(1);
        input_poll(&b, &x[0], &x[1], &x[2], &x[3]);
        CHECK(b == 0, "the open menu blocks the game's input");
        input_menu(0);
        input_poll(&b, &x[0], &x[1], &x[2], &x[3]);
        CHECK(b == 0, "the button held at the close stays ignored");
        SDL_SetJoystickVirtualButton(vj[0], SDL_GAMEPAD_BUTTON_EAST, true);
        SDL_UpdateJoysticks();
        input_poll(&b, &x[0], &x[1], &x[2], &x[3]);
        CHECK(__builtin_popcountll(b) == 1, "another button reaches the game while the first is still held");
        SDL_SetJoystickVirtualButton(vj[0], SDL_GAMEPAD_BUTTON_SOUTH, false);
        SDL_SetJoystickVirtualButton(vj[0], SDL_GAMEPAD_BUTTON_EAST, false);
        SDL_UpdateJoysticks();
        input_poll(&b, &x[0], &x[1], &x[2], &x[3]);
        SDL_SetJoystickVirtualButton(vj[0], SDL_GAMEPAD_BUTTON_SOUTH, true);
        SDL_UpdateJoysticks();
        input_poll(&b, &x[0], &x[1], &x[2], &x[3]);
        CHECK(b != 0, "after the release the same button works again");
        SDL_SetJoystickVirtualButton(vj[0], SDL_GAMEPAD_BUTTON_SOUTH, false);
        SDL_UpdateJoysticks();
    }
    steam_hidden_check(base);
    plat_setenv("MKW_PAD_P2", "1");
    CHECK(!input_pad_swap(0, 1) && !input_pad_can_move_up(1), "MKW_PAD_P2: player 1 keeps the keyboard");
    CHECK(input_pad_swap(1, 2) && input_pad_swap(1, 2), "MKW_PAD_P2: players 2 and 3 can trade");
    printf("probe-pads: %s\n", failures ? "FAILED" : "passed");
    return failures != 0;
}
